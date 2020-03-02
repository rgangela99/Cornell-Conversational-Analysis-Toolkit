from convokit import Corpus, CorpusObject
from typing import Callable
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pandas as pd
from scipy.sparse import vstack
from .classifier import Classifier


class BoWClassifier(Classifier):
    """
    Transformer that trains a classifier on the Corpus objects' text vector representation (e.g. bag-of-words, TF-IDF, etc)

    Runs on the Corpus's Users, Utterances, or Conversations (as specified by obj_type)

    Inherits from `Classifier` and has access to its methods.

    :param obj_type: "user", "utterance", or "conversation"
    :param vector_name: the metadata key where the Corpus object text vector is stored
    :param labeller: a (lambda) function that takes a Corpus object and returns True (y=1) or False (y=0) - i.e. labeller defines the y value of the object for fitting
    :param clf: a sklearn Classifier. By default, clf is a Pipeline with StandardScaler and LogisticRegression
    :param clf_feat_name: the metadata key to store the classifier prediction value under; default: "prediction"
    :param clf_prob_feat_name: the metadata key to store the classifier prediction score under; default: "pred_score"
    """
    def __init__(self, obj_type: str, vector_name="bow_vector",
                 labeller: Callable[[CorpusObject], bool] = lambda x: True,
                 clf=None, clf_feat_name: str = "prediction", clf_prob_feat_name: str = "pred_score"):
        if clf is None:
            print("Initializing default classification model (standard scaled logistic regression)")
            clf = Pipeline([("standardScaler", StandardScaler(with_mean=False)),
                            ("logreg", LogisticRegression(solver='liblinear'))])

        self.obj_type = obj_type
        self.labeller = labeller
        self.clf = clf
        self.clf_feat_name = clf_feat_name
        self.clf_prob_feat_name = clf_prob_feat_name
        self.vector_name = vector_name

    def fit(self, corpus: Corpus, y=None, selector: Callable[[CorpusObject], bool] = lambda x: True):
        """
        Fit the Transformer's internal classifier model on the Corpus objects, with an optional selector that filters for objects to be fit on.

        :param corpus: the target Corpus
        :param selector: a (lambda) function that takes a Corpus object and returns True or False (i.e. include /
        exclude). By default, the selector includes all objects of the specified type in the Corpus.
        :return: the fitted BoWClassifier
        """
        # collect texts for vectorization
        X = []
        y = []
        for obj in corpus.iter_objs(self.obj_type, selector):
            X.append(obj.meta[self.vector_name])
            y.append(self.labeller(obj))
        X = vstack(X)
        self.clf.fit(X, y)
        return self

    def transform(self, corpus: Corpus, selector: Callable[[CorpusObject], bool] = lambda x: True) -> Corpus:
        """
        Annotate the corpus objects with the classifier prediction and prediction score, with an optional selector
        that filters for objects to be classified. Objects that are not selected will get a metadata value of 'None'
        instead of the classifier prediction.

        :param corpus: the target Corpus
        :param selector: a (lambda) function that takes a Corpus object and returns True or False (i.e. include /
        exclude). By default, the selector includes all objects of the specified type in the Corpus.

        :return: the target Corpus annotated
        """
        objs = []
        X = []
        for obj in corpus.iter_objs(self.obj_type):
            if selector(obj):
                objs.append(obj)
                X.append(obj.meta[self.vector_name])
            else:
                obj.add_meta(self.clf_feat_name, None)
                obj.add_meta(self.clf_prob_feat_name, None)
        X = vstack(X)
        clfs, clfs_probs = self.clf.predict(X), self.clf.predict_proba(X)[:, 1]

        for idx, (clf, clf_prob) in enumerate(list(zip(clfs, clfs_probs))):
            obj = objs[idx]
            obj.add_meta(self.clf_feat_name, clf)
            obj.add_meta(self.clf_prob_feat_name, clf_prob)
        return corpus

    def fit_transform(self, corpus: Corpus, y=None, selector: Callable[[CorpusObject], bool] = lambda x: True) -> Corpus:
        self.fit(corpus, selector=selector)
        return self.transform(corpus, selector=selector)

    def summarize(self, corpus: Corpus, selector: Callable[[CorpusObject], bool] = lambda x: True):
        """
        Generate a DataFrame indexed by object id with the classifier predictions and scores

        :param corpus: the annotated Corpus
        :param selector: a (lambda) function that takes a Corpus object and returns True or False (i.e. include / exclude).
		By default, the selector includes all objects of the specified type in the Corpus.
        :return: a pandas DataFrame
        """
        objId_clf_prob = []

        for obj in corpus.iter_objs(self.obj_type, selector):
            objId_clf_prob.append((obj.id, obj.meta[self.clf_feat_name], obj.meta[self.clf_prob_feat_name]))

        return pd.DataFrame(list(objId_clf_prob),
                           columns=['id', self.clf_feat_name, self.clf_prob_feat_name])\
                        .set_index('id').sort_values(self.clf_prob_feat_name, ascending=False)

    # def evaluate_with_cv(self, corpus: Corpus = None,
    #                      objs: List[CorpusObject] = None, cv=KFold(n_splits=5)):
    #     raise NotImplementedError
    #
    # def evaluate_with_train_test_split(self, corpus: Corpus = None,
    #              objs: List[CorpusObject] = None,
    #              test_size: float = 0.2):
    #     raise NotImplementedError




