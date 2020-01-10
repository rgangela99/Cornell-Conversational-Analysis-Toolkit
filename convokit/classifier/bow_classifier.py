from convokit import Corpus, CorpusObject, Transformer
from typing import Callable, List
from sklearn.feature_extraction.text import CountVectorizer as CV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
import pandas as pd
from .util import get_coefs_helper
from scipy.sparse import vstack
from convokit.bag_of_words import BoWTransformer
from .classifier import Classifier

class BoWClassifier(Classifier):
    def __init__(self, obj_type: str, vector_name="bow_vector",
                 labeller: Callable[[CorpusObject], bool] = lambda x: True,
                 selector: Callable[[CorpusObject], bool] = lambda x: True,
                 clf=None, clf_feat_name: str = "prediction", clf_prob_feat_name: str = "pred_score"):

        if clf is None:
            print("Initializing default classification model (standard scaled logistic regression)")
            clf = Pipeline([("standardScaler", StandardScaler(with_mean=False)),
                            ("logreg", LogisticRegression(solver='liblinear'))])

        self.obj_type = obj_type
        self.labeller = labeller
        self.selector = selector
        self.clf = clf
        self.clf_feat_name = clf_feat_name
        self.clf_prob_feat_name = clf_prob_feat_name
        self.vector_name = vector_name

    def fit(self, corpus: Corpus, y=None):
        # collect texts for vectorization
        X = []
        y = []
        for obj in corpus.iter_objs(self.obj_type, self.selector):
            X.append(obj.meta[self.vector_name])
            y.append(self.labeller(obj))
        X = vstack(X)
        self.clf.fit(X, y)
        return self

    def transform(self, corpus: Corpus) -> Corpus:
        objs = []
        X = []
        for obj in corpus.iter_objs(self.obj_type):
            if self.selector(obj):
                objs.append(obj)
                X.append(obj.meta[self.vector_name])
            else:
                obj.meta[self.vector_name] = None
        X = vstack(X)
        clfs, clfs_probs = self.clf.predict(X), self.clf.predict_proba(X)[:, 1]

        for idx, (clf, clf_prob) in enumerate(list(zip(clfs, clfs_probs))):
            obj = objs[idx]
            obj.add_meta(self.clf_feat_name, clf)
            obj.add_meta(self.clf_prob_feat_name, clf_prob)
        return corpus

    def summarize(self, corpus: Corpus, use_selector=True):
        objId_clf_prob = []

        for obj in corpus.iter_objs(self.obj_type, self.selector if use_selector else lambda _: True):
            objId_clf_prob.append((obj.id, obj.meta[self.clf_feat_name], obj.meta[self.clf_prob_feat_name]))

        return pd.DataFrame(list(objId_clf_prob),
                           columns=['id', self.clf_feat_name, self.clf_prob_feat_name])\
                        .set_index('id').sort_values(self.clf_prob_feat_name, ascending=False)

    def evaluate_with_cv(self, corpus: Corpus = None,
                         objs: List[CorpusObject] = None, cv=KFold(n_splits=5)):
        raise NotImplementedError

    def evaluate_with_train_test_split(self, corpus: Corpus = None,
                 objs: List[CorpusObject] = None,
                 test_size: float = 0.2):
        raise NotImplementedError




