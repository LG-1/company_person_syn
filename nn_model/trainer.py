import pandas as pd
import numpy as np
import pickle
import time

import torch
import torch.utils.data
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


class Trainer(object):
    def factory(algorithm, params):
        if algorithm == "lstm":
            return LSTMTrainer(params)

        raise AssertionError("Bad trainer creation: " + algorithm)

    factory = staticmethod(factory)



class LSTMTrainer(object):
    def __init__(self, params):
        self.class_point = 0.5
        self.a = 0
        self.b = 0

        self.max_features = params.get("max_features", 1000)
        self.max_len = params.get("max_len", 30)
        self.batch_size = params.get("batch_size", 512)
        self.epochs = params.get("epochs", 4)
        self.embed_size = params.get("embed_size", 200)
        self.test_size = params.get("test_size", 0.3)
        self.random_state = params.get("random_state", 123)
        self.model_type = params.get("model_type", "nn_comment_review")

        self.model = None
        self.tokenizer = None
        self.bins = None

    def gen_tokenizer(self, x_g):
        self.tokenizer = Tokenizer(num_words=self.max_features, lower=False)
        for x in x_g:
            x = x.fillna('nan')
            print(f'tokenizer i: {i} ')
            x = check_x_type(x)
            self.tokenizer.fit_on_texts(x.flatten().tolist())

        print(
            f'total tokenizer word index len: {len(self.tokenizer.word_index)}')

    def fit(self, x_g, y_g):
        word_index = self.tokenizer.word_index
        emb = "load w2v"
        embedding_matrix, self.embed_size = embedding_matrix_p(
            emb, word_index, self.max_features)
        self.model = nn_model(self.model_type, self.max_len,
                              embedding_matrix, self.embed_size, self.max_features)

        loss_fn = torch.nn.BCEWithLogitsLoss(reduction='sum')
        optimizer = torch.optim.Adam(self.model.parameters())

        for epoch in range(self.epochs):
            start_time = time.time()
            self.model.train()
            avg_loss = 0.

            all_val_x = []
            all_val_y = []
            import itertools
            x_g, x_g2 = itertools.tee(x_g)
            y_g, y_g2 = itertools.tee(y_g)

            for i, (x, y) in enumerate(zip(x_g, y_g)):
                print(f"{i+1} of epochs {epoch+1}: {i+1}/{epoch+1}/{self.epochs}")
                x = x.fillna('nan')
                x = check_x_type(x)
                y = check_x_type(y)
                y = np.array(y, dtype=float)

                x = seq_pad_x(x, self.tokenizer, self.max_len)

                x, val_x, y, val_y = train_test_split(x, y,
                                                      test_size=self.test_size, random_state=self.random_state)

                trn_loader = prepare_nn_loader(
                    x, y, self.model_type, self.batch_size, shuffle=True)
                all_val_x.extend(val_x)
                all_val_y.extend(val_y)

                for *x_batch, y_batch in trn_loader:
                    x = prepare_nn_x(x_batch, self.model_type)
                    y_pred = self.model(x)

                    loss = loss_fn(y_pred, y_batch)
                    optimizer.zero_grad()
                    loss.backward()

                    optimizer.step()
                    avg_loss += loss.item() / len(trn_loader)
            avg_loss /= (i+1)

            self.model.eval()

            val_loader = prepare_nn_loader(np.stack(all_val_x), np.stack(all_val_y),
                                           self.model_type, self.batch_size, shuffle=False)
            avg_val_loss = 0.
            valid_preds = np.zeros(len(val_loader.dataset))
            for i, (*x_batch, y_batch) in enumerate(val_loader):
                x = prepare_nn_x(x_batch, self.model_type)
                y_pred = self.model(x).detach()

                avg_val_loss += loss_fn(y_pred,
                                        y_batch).item() / len(val_loader)
                valid_preds[i * self.batch_size:(i + 1) *
                            self.batch_size] = sigmoid(y_pred.numpy())[:, 0]

            x_g2, x_g = itertools.tee(x_g2)
            y_g2, y_g = itertools.tee(y_g2)

            elapsed_time = time.time() - start_time

            print('Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t time={:.2f}s'.format(epoch + 1, self.epochs, avg_loss,
                                                                                         avg_val_loss, elapsed_time))

        train_preds = []
        true_ys = []
        for x, y in zip(x_g, y_g):
            x = x.fillna('nan')
            x = check_x_type(x)
            y = check_x_type(y)
            y = np.array(y, dtype=float)

            x = seq_pad_x(x, self.tokenizer, self.max_len)
            trn_loader = prepare_nn_loader(
                x, y, self.model_type, self.batch_size, shuffle=True)
            train_pred = np.zeros(len(trn_loader.dataset))
            true_y = np.zeros(len(trn_loader.dataset))
            for i, (*x_batch, y_batch) in enumerate(trn_loader):
                x = prepare_nn_x(x_batch, self.model_type)
                y_pred = self.model(x).detach()
                train_pred[i * self.batch_size:(i + 1) *
                           self.batch_size] = sigmoid(y_pred.numpy())[:, 0]
                true_y[i * self.batch_size:(i + 1) *
                       self.batch_size] = y_batch.numpy().reshape(-1,)

            train_preds.extend(train_pred)
            true_ys.extend(true_y)

        train_preds = np.array(train_preds)
        true_ys = np.array(true_ys)

        self.class_point = get_best_thresh(true_ys, train_preds)
        self.class_point = sorted(
            train_preds)[-int(sum(true_ys))]  # 依据数据分布比例更新分类点
        self.bins = self.decode_bins(train_preds, 10)

    def predict(self, x_g):
        preds = []
        for x in x_g:
            x = x.fillna('nan')
            x = check_x_type(x)
            # 预测前需要做的文本序列化-对齐 操作
            x = seq_pad_x(x, self.tokenizer, self.max_len)

            x_loader = prepare_nn_loader_pre(x, self.model_type)
            pred = np.zeros(len(x_loader.dataset))

            for i, (*x_batch, ) in enumerate(x_loader):
                x = prepare_nn_x(x_batch, self.model_type)

                y_pred = self.model(x).detach()
                pred[i * self.batch_size:(i + 1) *
                     self.batch_size] = sigmoid(y_pred.numpy())[:, 0]

            preds.extend(pred)
        preds = np.array(preds)
        return (preds > self.class_point).astype(int)

    def predict_proba(self, x_g):
        preds = []
        for x in x_g:
            x = x.fillna('nan')
            x = check_x_type(x)
            # 预测前需要做的文本序列化-对齐 操作
            x = seq_pad_x(x, self.tokenizer, self.max_len)

            x_loader = prepare_nn_loader_pre(x, self.model_type)
            pred = np.zeros(len(x_loader.dataset))

            for i, (*x_batch,) in enumerate(x_loader):
                x = prepare_nn_x(x_batch, self.model_type)

                y_pred = self.model(x).detach()
                pred[i * self.batch_size:(i + 1) *
                     self.batch_size] = sigmoid(y_pred.numpy())[:, 0]

            preds.extend(pred)
        preds = np.array(preds)
        return preds

    def save_model(self, save_mode=None, model_path=None):

        if save_mode == "pkl":
            with open(model_path, "wb") as fout:
                pickle.dump(self, fout)
        return self.model

    @staticmethod
    def decode_bins(arr, n):
        # 序列数据的 n 等分点
        bins = [sorted(arr)[int(len(arr) * 0.1 * i)] for i in range(1, n)]
        return bins



class NeuralNet_Reply(nn.Module):
    def __init__(self, embedding_matrix, max_features, embed_size, maxlen):
        super(NeuralNet_Reply, self).__init__()

        hidden_size = 40

        self.embedding = nn.Embedding(max_features, embed_size)
        self.embedding.weight = nn.Parameter(
            torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False

        self.embedding_dropout = nn.Dropout2d(0.1)
        self.lstm = nn.LSTM(embed_size, hidden_size,
                            bidirectional=True, batch_first=True)
        self.gru = nn.GRU(hidden_size * 2, hidden_size,
                          bidirectional=True, batch_first=True)

        self.lstm_attention = Attention(hidden_size * 2, maxlen)
        self.gru_attention = Attention(hidden_size * 2, maxlen)

        self.linear1 = nn.Linear(640, 128)
        self.linear2 = nn.Linear(128, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.out = nn.Linear(64, 1)

    def forward(self, x):
        h_embedding = self.embedding(x[0])
        h_embedding = torch.squeeze(
            self.embedding_dropout(torch.unsqueeze(h_embedding, 0)))

        h_lstm, _ = self.lstm(h_embedding)
        h_gru, _ = self.gru(h_lstm)

        h_lstm_atten = self.lstm_attention(h_lstm)
        h_gru_atten = self.gru_attention(h_gru)

        # global average pooling
        avg_pool = torch.mean(h_gru, 1)
        # global max pooling
        max_pool, _ = torch.max(h_gru, 1)

        h_embedding2 = self.embedding(x[1])
        h_embedding2 = torch.squeeze(
            self.embedding_dropout(torch.unsqueeze(h_embedding2, 0)))

        h_lstm2, _ = self.lstm(h_embedding2)
        h_gru2, _ = self.gru(h_lstm2)

        h_lstm_atten2 = self.lstm_attention(h_lstm2)
        h_gru_atten2 = self.gru_attention(h_gru2)

        # global average pooling
        avg_pool2 = torch.mean(h_gru2, 1)
        # global max pooling
        max_pool2, _ = torch.max(h_gru2, 1)

        conc = torch.cat(
            (h_lstm_atten, h_gru_atten, avg_pool, max_pool, h_lstm_atten2, h_gru_atten2, avg_pool2, max_pool2), 1)
        conc = self.relu(self.linear1(conc))
        conc = self.dropout(conc)
        conc = self.relu(self.linear2(conc))
        conc = self.dropout(conc)
        out = self.out(conc)

        return out



def train_model_single(model, model_type, train_loader, valid_loader, batch_size, n_epochs):

    loss_fn = torch.nn.BCEWithLogitsLoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(n_epochs):
        start_time = time.time()
        model.train()
        avg_loss = 0.

        for *x_batch, y_batch in train_loader:
            x = prepare_nn_x(x_batch, model_type)
            y_pred = model(x)

            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            avg_loss += loss.item() / len(train_loader)

        model.eval()

        avg_val_loss = 0.
        valid_preds = np.zeros(len(valid_loader.dataset))
        for i, (*x_batch, y_batch) in enumerate(valid_loader):
            x = prepare_nn_x(x_batch, model_type)
            y_pred = model(x).detach()

            avg_val_loss += loss_fn(y_pred, y_batch).item() / len(valid_loader)
            valid_preds[i * batch_size:(i + 1) *
                        batch_size] = sigmoid(y_pred.numpy())[:, 0]

        elapsed_time = time.time() - start_time

        print('Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t time={:.2f}s'.format(epoch + 1, n_epochs, avg_loss,
                                                                                     avg_val_loss, elapsed_time))

    # 输出训练数据预测结果
    train_pred = []
    train_pred = np.zeros(len(train_loader.dataset))
    for i, (*x_batch, y_batch) in enumerate(train_loader):
        x = prepare_nn_x(x_batch, model_type)
        y_pred = model(x).detach()
        train_pred[i * batch_size:(i + 1) *
                   batch_size] = sigmoid(y_pred.numpy())[:, 0]

    return model, train_pred


def nn_model(model_type, max_len, embedding_matrix, embed_size, max_features):
    model = None
    if model_type == "nn_reply_review":
        model = NeuralNet_Reply(
            embedding_matrix, max_features, embed_size, max_len)

    return model


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def get_cats(x, bins):
    # 返回x处在第几个分箱
    if x < bins[0]:
        return 1
    if x > bins[-1]:
        return len(bins) + 1
    for i, k in enumerate(bins[:-1]):
        if (x > bins[i]) and (x <= bins[i+1]):
            return i + 2


def check_x_type(x):
    if isinstance(x, pd.DataFrame):
        x = x.values
    if isinstance(x, pd.Series):
        x = x.values
    return x


def seq_pad_x(x, tokenizer, max_len):
    # 序列化并展平数据
    x_o = x.copy()
    temp = np.random.normal(0, 0, (x_o.shape[0], x_o.shape[1], max_len))
    for i in range(x_o.shape[1]):
        x_o[:, i] = tokenizer.texts_to_sequences(x_o[:, i])
        temp[:, i] = pad_sequences(x_o[:, i], maxlen=max_len)
    del x_o, x

    return temp


def prepare_nn_loader(x, y, model_type, batch_size=512, shuffle=False, device=None):
    # 准备X数据格式
    if model_type == "nn_comment_review":
        x_x = torch.tensor(x[:, 0], dtype=torch.long)
        y_y = torch.tensor(y[:, np.newaxis], dtype=torch.float32)
        train = torch.utils.data.TensorDataset(x_x, y_y)

        loader = torch.utils.data.DataLoader(
            train, batch_size=batch_size, shuffle=shuffle)

    elif model_type == "nn_reply_review":
        x_x = torch.tensor(x[:, 0], dtype=torch.long)
        r_x = torch.tensor(x[:, 1], dtype=torch.long)
        y_y = torch.tensor(y[:], dtype=torch.float32)
        train = torch.utils.data.TensorDataset(x_x, r_x, y_y)

        loader = torch.utils.data.DataLoader(
            train, batch_size=batch_size, shuffle=shuffle)


    return loader


def prepare_nn_loader_pre(x, model_type, batch_size=512, shuffle=False):
    # 准备X数据格式
    if model_type == "nn_comment_review":
        x_x = torch.tensor(x[:, 0], dtype=torch.long)
        train = torch.utils.data.TensorDataset(x_x)

        loader = torch.utils.data.DataLoader(
            train, batch_size=batch_size, shuffle=shuffle)

    elif model_type == "nn_reply_review":
        x_x = torch.tensor(x[:, 0], dtype=torch.long)
        r_x = torch.tensor(x[:, 1], dtype=torch.long)
        train = torch.utils.data.TensorDataset(x_x, r_x)

        loader = torch.utils.data.DataLoader(
            train, batch_size=batch_size, shuffle=shuffle)

    return loader


def prepare_nn_x(x, model_type='nn_comment_review'):
    # 准备X数据格式
    return x


def get_best_thresh(y, pred_y):
    thresholds = []
    for thresh in np.arange(0.2, 0.901, 0.01):
        thresh = np.round(thresh, 2)
        res = f1_score(y, (pred_y > thresh).astype(int))
        thresholds.append([thresh, res])
        print("Val data F1 score at threshold {0} is {1}".format(thresh, res))
    thresholds.sort(key=lambda x: x[1], reverse=True)
    best_thresh = thresholds[0][0]
    return best_thresh


def embedding_matrix_p(emb, word_index, max_features, mode="zero"):
    nb_words = min(max_features, len(word_index))
    if mode == "zero":
        embed_size = len(emb.emb("a"))
        embedding_matrix = np.random.normal(0, 0, (nb_words, embed_size))
        for word, i in word_index.items():
            if i >= max_features:
                continue
            embedding_matrix[i] = np.array(emb.emb(word))

        return embedding_matrix, embed_size


class Attention(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super(Attention, self).__init__(**kwargs)

        self.supports_masking = True

        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.features_dim = 0

        weight = torch.zeros(feature_dim, 1)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight)

        if bias:
            self.b = nn.Parameter(torch.zeros(step_dim))

    def forward(self, x, mask=None):
        feature_dim = self.feature_dim
        step_dim = self.step_dim

        eij = torch.mm(
            x.contiguous().view(-1, feature_dim),
            self.weight
        ).view(-1, step_dim)

        if self.bias:
            eij = eij + self.b

        eij = torch.tanh(eij)
        a = torch.exp(eij)

        if mask is not None:
            a = a * mask

        a = a / torch.sum(a, 1, keepdim=True) + 1e-10

        weighted_input = x * torch.unsqueeze(a, -1)
        return torch.sum(weighted_input, 1)


def solve_zoom_formula(class_point):
    """
    this function will solve "y = a*x^2+b*x" for zoom value base on class point
    :param class_point:
    :return:
    """
    # slove: y = a*x^2+b*x
    #
    A = np.array([[1, 1], [class_point * class_point, class_point]])
    b1 = np.array([1, 0.5])
    r = np.linalg.solve(A, b1)
    a, b = r
    return a, b


def get_zoom_result(a, b, x):
    return max(min(a * x ** 2 + b * x, 1), 0)
