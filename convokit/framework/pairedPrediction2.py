from .framework import Framework
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.model_selection import train_test_split
from typing import List, Union
from pandas import DataFrame

class PairedPrediction2(Framework):
    def __init__(self, pairing_func, label_func, filter_func, datatype="conversation"):
        self.pairing_func = pairing_func
        self.label_func = label_func
        self.filter_func = filter_func
        assert datatype in {'conversation', 'utterance', 'user'}
        self.datatype = datatype

    def fit(self, corpus):
        iter_objs = {'conversation': corpus.iter_conversations,
                 'utterance': corpus.iter_utterances,
                 'user': corpus.iter_users
                 }

        filtered_objs = [obj for obj in iter_objs[self.datatype]() if self.filter_func(obj)]

