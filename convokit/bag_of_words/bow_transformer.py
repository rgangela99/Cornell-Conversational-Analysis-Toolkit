from convokit import Corpus, CorpusObject, Transformer
from typing import Callable, List
from sklearn.feature_extraction.text import CountVectorizer as CV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pandas as pd

class BoWTransformer(Transformer):
    def __init__(self, obj_type: str, vectorizer=None, vector_name="bow_vector",
                 text_func: Callable[[CorpusObject], str] = lambda utt: utt.text,
                 selector: Callable[[CorpusObject], bool] = lambda x: True):
        if vectorizer is None:
            print("Initializing default unigram CountVectorizer...")
            self.vectorizer = CV(decode_error='ignore', min_df=10, max_df=.5,
                                 ngram_range=(1, 1), binary=False, max_features=15000)
        else:
            self.vectorizer = vectorizer

        self.obj_type = obj_type
        self.vector_name = vector_name
        self.text_func = text_func
        self.selector = selector

    def fit(self, corpus: Corpus, y=None):
        # collect texts for vectorization
        docs = []
        for obj in corpus.iter_objs(self.obj_type, self.selector):
            docs.append(self.text_func(obj))

        self.vectorizer.fit(docs)

    def transform(self, corpus: Corpus) -> Corpus:
        for obj in corpus.iter_objs(self.obj_type):
            if self.selector(obj):
                obj.meta[self.vector_name] = self.vectorizer.transform([self.text_func(obj)])
            else:
                obj.meta[self.vector_name] = None

        return corpus

    def get_vocabulary(self):
        return self.vectorizer.vocabulary_