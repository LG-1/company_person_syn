import json
import re
import pandas as pd
import numpy as np
from collections import Counter
import distance
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.linalg import norm

nn_reply_review = ["question_text_format", "reply_text_format"]
nn_reply_review_all = ["question_text_format", "reply_text_format"]
nn_reply_review_columns = ["query", "reply"]

class FeatureCreation(object):
    def __init__(self, input_df, feature_type="nn_reply_review", *args):
        self.features = nn_reply_review
        self.all_ = nn_reply_review_all
        # create feature need reference all data
        self.c = {}
        self.columns=nn_reply_review_columns
        self.input_df = input_df
        self.feature_type = feature_type
        if args:
            self.uid = list(args)[0]
        else:
            self.uid = None


    def create_ft_mx(self, user_profile=None):

        if self.feature_type == "nn_reply_review":
            self.input_df[self.features] = self.input_df.apply(
                lambda x: self.__single_ft_mx(
                    self.feature_type, self.features, x.query, x.reply
                ),
                axis=1,
            )
            return self.input_df[self.features]


    def __single_ft_mx(self, feature_type="home_rc_r", features=None, *args):
        

        if feature_type == "nn_reply_review":
            words = str(args[0])
            reply = str(args[1])

            kwargs = {
                "words": words,
                "reply": reply,
            }
            return self.__create_featrues(features, **kwargs)


    def __create_featrues(self, features, **kwargs):
        data_dict = {}
        for feature in features:

            if feature == "lstm_text_format":
                data_dict[feature] = self.__lstm_text_format(
                    kwargs.get("segmented_words")
                )
            if feature == "question_text_format":
                data_dict[feature] = self.__lstm_text_format(
                    kwargs.get("segmented_words")
                )
            if feature == "reply_text_format":
                data_dict[feature] = self.__lstm_text_format(
                    kwargs.get("segmented_reply")
                )

        return pd.DataFrame.from_dict(data_dict, orient="index").T.loc[0, ]

    def __lstm_text_format(self, segmented_words):
        # words_str = " ".join([part["word"] for part in segmented_words])
        words_str = " ".join(segmented_words)
        return words_str
