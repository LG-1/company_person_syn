from sklearn.utils import check_array
from sklearn.svm import SVC
from sklearn.metrics.pairwise import pairwise_kernels
import pandas as pd
import numpy as np
import json

from ml.Trainer import prepare_nn_loader_pre, sigmoid
from ml.Trainer import prepare_nn_loader
from ml.Trainer import check_x_type, seq_pad_x, prepare_nn_x
from ml.Trainer import Trainer
from how.ml.MathUtil import cosine_
from ml.FeaturesCreation import FeatureCreation


class Predictor(object):
    def factory(algorithm, model):
        if algorithm == "lstm":
            return LSTMPredictor(model)

        raise AssertionError("Bad car creation: " + algorithm)

    factory = staticmethod(factory)


class LSTMPredictor(object):
    def __init__(self, model):

        self.tokenizer = model.tokenizer
        self.max_len = model.max_len
        self.class_point = model.class_point
        self.model_type = model.model_type
        self.batch_size = model.batch_size
        self.a = model.a
        self.b = model.b
        self.model = model.model
        self.bins = model.bins

    def predict(self, x):
        alone_flag = False
        if x.shape[0] == 1:
            alone_flag = True
            x = x.append(x)

        x = check_x_type(x)
        x = seq_pad_x(x, self.tokenizer, self.max_len)  # 预测前需要做的文本序列化-对齐 操作

        x_loader = prepare_nn_loader_pre(x, self.model_type)
        preds = np.zeros(len(x_loader.dataset))

        for i, (*x_batch, ) in enumerate(x_loader):
            x = prepare_nn_x(x_batch, self.model_type)

            y_pred = self.model(x).detach()
            preds[i * self.batch_size:(i + 1) *
                  self.batch_size] = sigmoid(y_pred.numpy())[:, 0]

            # preds = np.array([get_zoom_result(self.a, self.b, i) for i in preds])

        if alone_flag:
            preds = np.array([preds[0]])
        return (preds > self.class_point).astype(int)

    def predict_proba(self, x):
        alone_flag = False
        if x.shape[0] == 1:
            alone_flag = True
            x = x.append(x)

        x = check_x_type(x)
        x = seq_pad_x(x, self.tokenizer, self.max_len)  # 预测前需要做的文本序列化-对齐 操作

        x_loader = prepare_nn_loader_pre(x, self.model_type)
        preds = np.zeros(len(x_loader.dataset))

        for i, (*x_batch,) in enumerate(x_loader):
            x = prepare_nn_x(x_batch, self.model_type)
            y_pred = self.model(x).detach()
            preds[i * self.batch_size:(i + 1) *
                  self.batch_size] = sigmoid(y_pred.numpy())[:, 0]

        if alone_flag:
            preds = np.array([preds[0]])
        return preds


