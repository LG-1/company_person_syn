from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    recall_score,
    f1_score,
)
from sklearn.utils import shuffle
from ml.Trainer import Trainer
from ml.FeaturesCreation import FeatureCreation
from django.conf import settings
import json
import pandas as pd
import numpy as np
import sys
import ipdb
import itertools


class CrossValidation(object):
    def __init__(
        self, df_url, feature_type, algorithms, label_name, cached, proba_ratio, *args, **kwargs
    ):
        self.df_url = df_url
        self.feature_type = feature_type
        self.algorithms = algorithms
        self.label_name = label_name
        self.cached = cached
        self.proba_ratio = proba_ratio
        self.do_train = kwargs.get('do_train', 1)
        self.do_valid = kwargs.get('do_valid', 1)

    def __single_train_by_al(self, al, x, y):
        trainer = Trainer.factory(
            algorithm=al, params=json.loads(
                """
                {"max_features": 3000, "max_len": 99, "batch_size": 512, "epochs": 5, "test_size": 0.05, "random_state":123, "model_type": "nn_reply_review"}
                """
        )
        # print(x.shape)  # to see to check shape data
        # print(x.columns)  # to see to check features

        x, x2 = itertools.tee(x)
        trainer.gen_tokenizer(x)
        print("trainer tokenizer generate completed!")
        x2, x = itertools.tee(x2)

        if self.proba_ratio:
            trainer.fit(x, y)
            result = trainer.predict_proba(x, self.proba_ratio)  # 选取特定的分类点
            result = (result > 0.5).astype(np.int)
        else:
            trainer.fit(x, y)
            x2, x = itertools.tee(x2)
            result = trainer.predict(x)
        return result, trainer

    def __single_train_by_data(self):
        pass

    def cross_validate(self, validate_type="algorithm", metrics=["accuracy"]):
        # when debug something else, you had better cache the X,
        # because it may cost much time to create it when have a lot features.

        if self.cached:
            X = pd.read_csv("notebooks/cached/" + self.feature_type +
                            "/cached_X.txt", sep='\t', index_col=False, chunksize=600)
            Y = pd.read_csv("notebooks/cached/" + self.feature_type +
                            "/cached_Y.txt", sep='\t', index_col=False, chunksize=600)

        else:
            input_df = pd.read_csv(
                self.df_url, index_col=False, sep="\t").fillna(0).head(1500)
            fc = FeatureCreation(
                input_df=input_df, feature_type=self.feature_type)
            X = fc.create_ft_mx()
            X = pd.DataFrame(X)
            X = X.fillna('nan')
            X.to_csv("notebooks/cached/" + self.feature_type +
                     "/cached_X.txt", sep='\t', index=False)

            if self.label_name:
                Y = input_df[[self.label_name]]
                Y.to_csv("notebooks/cached/" + self.feature_type +
                         "/cached_Y.txt", sep='\t', index=False)

            X = pd.read_csv("notebooks/cached/" + self.feature_type +
                            "/cached_X.txt", sep='\t', index_col=False, chunksize=600)
            Y = pd.read_csv("notebooks/cached/" + self.feature_type +
                            "/cached_Y.txt", sep='\t', index_col=False, chunksize=600)

        Y, Y2 = itertools.tee(Y)
        X, X2 = itertools.tee(X)
        y_ture = []
        for y in Y2:
            y_ture.extend(np.array(y, dtype=float))
        y_ture = np.stack(y_ture).reshape(-1,)

        Y, Y2 = itertools.tee(Y)

        if validate_type == "algorithm":
            for al in self.algorithms:
                input_df = pd.DataFrame()
                model_path = ("ml/sources/" + self.feature_type +
                              "/" + al + "/" + al + "model.pkl")
                if self.do_train:
                    result, model = self.__single_train_by_al(al, X, Y)
                    X2, X = itertools.tee(X2)
                    Y2, Y = itertools.tee(Y2)

                    input_df[al] = result  # 训练结果保存，用于验证查看实际数据上的表现
                    # # 暂时先这样写，后期模型的保存再更新 #########
                    model.save_model(save_mode="pkl", model_path=model_path)

                    print("===================================== " + al + " ======")
                    for metric in metrics:
                        self.__print_validation_result(y_ture, result, metric)

                if self.do_valid:
                    with open(model_path, 'rb') as fin:
                        import pickle
                        model = pickle.load(fin)
                        input_df[al + "_proba"] = model.predict_proba(X)
                    X2, X = itertools.tee(X2)
                input_df.to_csv(
                    "notebooks/cached/" + self.feature_type + "/output_" + al + ".csv", index=False)

    @staticmethod
    def __print_validation_result(true, prediction, metric):
        if metric == "accuracy":
            print("Accuracy score : ", accuracy_score(true, prediction))
        if metric == "f1_score":
            print("F1 Score: ", f1_score(true, prediction))
        if metric == "recall":
            print("Recall score   : ", recall_score(true, prediction))
        if metric == "class_report":
            print("classification report : \n",
                  classification_report(true, prediction))
        if metric == "confusion_matrix":
            confusion_m = confusion_matrix(true, prediction)
            print(
                "confusion_matrix : \n",
                pd.DataFrame(confusion_m, columns=["pre_0", "pre_1"], index=["true_0", "true_1"]))


if __name__ == "__main__":

    def run_mode(mode="words_meaning"):
        if mode == "nn_reply_review":
            df_url = "notebooks/03_reply_review/data/input/edit_reserve_for_lg.txt"
            cv = CrossValidation(
                df_url=df_url,
                # user_profile=None,
                feature_type="nn_reply_review",
                label_name="label",
                algorithms=[sys.argv[2]],  # lstm
                cached=int(sys.argv[3]),
                proba_ratio=False,
            )
            cv.cross_validate("algorithm", [
                              "f1_score", "accuracy", "recall", "confusion_matrix", "class_report"])

        if mode == "nn_reply_review_batch":
            data_url = "notebooks/19_baidu_zhidao_corpus/output/zhidao_reply_review_sample.txt"
            data = pd.read_csv(data_url, index_col=False, sep="\t").fillna(0)
            input_df = data[data.label.isin([0, 1])]
            print(input_df.head())
            print(input_df.shape)
            cv = CrossValidation(
                input_df=input_df,
                # user_profile=None,
                feature_type="nn_reply_review",
                label_name="label",
                algorithms=[sys.argv[2]],  # lstm
                cached=int(sys.argv[3]),
                proba_ratio=False,
                do_train=0
            )
            cv.cross_validate("algorithm", [
                              "f1_score", "accuracy", "recall", "confusion_matrix", "class_report"])



    run_mode(mode=sys.argv[1])
    # sys.argv[1] 运行任务 eg:comment_review
    # sys.argv[2] 模型名称 eg: rfc
    # sys.argv[3] 是否已缓存数据 1/0
