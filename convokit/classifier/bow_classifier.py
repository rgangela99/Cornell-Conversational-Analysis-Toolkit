from convokit import Corpus, CorpusObject
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from typing import Callable, List
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
from scipy.sparse import vstack
from .classifier import Classifier
import numpy as np
from .util import extract_feats_and_label_bow

class BoWClassifier(Classifier):
    """
    Transformer that trains a classifier on the Corpus objects' text vector representation (e.g. bag-of-words, TF-IDF, etc)

    Runs on the Corpus's Users, Utterances, or Conversations (as specified by obj_type)

    Inherits from `Classifier` and has access to its methods.

    :param obj_type: "speaker", "utterance", or "conversation"
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
        self.vector_name = vector_name

        super().__init__(obj_type=obj_type, pred_feats=[], labeller=labeller,
                         clf=clf, clf_feat_name=clf_feat_name, clf_prob_feat_name=clf_prob_feat_name)

    def fit(self, corpus: Corpus, y=None, selector: Callable[[CorpusObject], bool] = lambda x: True):
        """
        Fit the Transformer's internal classifier model on the Corpus objects, with an optional selector that filters for objects to be fit on.

        :param corpus: the target Corpus
        :param selector: a (lambda) function that takes a Corpus object and returns True or False (i.e. include / exclude). By default, the selector includes all objects of the specified type in the Corpus.
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
        :param selector: a (lambda) function that takes a Corpus object and returns True or False (i.e. include / exclude). By default, the selector includes all objects of the specified type in the Corpus.

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

    def transform_objs(self, objs: List[CorpusObject]) -> List[CorpusObject]:
        """
        Run classifier on list of Corpus objects and annotate them with the predictions and prediction scores

        :param objs: list of Corpus objects

        :return: list of annotated Corpus objects
        """
        X, _ = extract_feats_and_label_bow(corpus=None, objs=objs, obj_type=None, vector_name=self.vector_name,
                                    labeller=self.labeller, selector=None)

        X = X.toarray()
        # obj_ids = [obj.id for obj in objs]
        clfs, clfs_probs = self.clf.predict(X), self.clf.predict_proba(X)[:, 1]

        for idx, (clf, clf_prob) in enumerate(list(zip(clfs, clfs_probs))):
            obj = objs[idx]
            obj.add_meta(self.clf_feat_name, clf)
            obj.add_meta(self.clf_prob_feat_name, clf_prob)

        return objs

    def fit_transform(self, corpus: Corpus, y=None, selector: Callable[[CorpusObject], bool] = lambda x: True) -> Corpus:
        self.fit(corpus, selector=selector)
        return self.transform(corpus, selector=selector)

    def summarize(self, corpus: Corpus, selector: Callable[[CorpusObject], bool] = lambda x: True):
        """
        Generate a DataFrame indexed by object id with the classifier predictions and scores

        :param corpus: the annotated Corpus
        :param selector: a (lambda) function that takes a Corpus object and returns True or False (i.e. include / exclude). By default, the selector includes all objects of the specified type in the Corpus.
        :return: a pandas DataFrame
        """
        objId_clf_prob = []

        for obj in corpus.iter_objs(self.obj_type, selector):
            objId_clf_prob.append((obj.id, obj.meta[self.clf_feat_name], obj.meta[self.clf_prob_feat_name]))

        return pd.DataFrame(list(objId_clf_prob),
                           columns=['id', self.clf_feat_name, self.clf_prob_feat_name])\
                        .set_index('id').sort_values(self.clf_prob_feat_name, ascending=False)

    def summarize_objs(self, objs: List[CorpusObject]):
        """
        Generate a pandas DataFrame (indexed by object id, with prediction and prediction score columns) of classification results.

        Runs on a list of Corpus objects.

        :param objs: list of Corpus objects
        :return: pandas DataFrame indexed by Corpus object id
        """
        objId_clf_prob = []
        for obj in objs:
            objId_clf_prob.append((obj.id, obj.meta[self.clf_feat_name], obj.meta[self.clf_prob_feat_name]))

        return pd.DataFrame(list(objId_clf_prob),
                            columns=['id', self.clf_feat_name, self.clf_prob_feat_name]).set_index('id').sort_values(self.clf_prob_feat_name)


    def evaluate_with_train_test_split(self, corpus: Corpus = None,
                                       objs: List[CorpusObject] = None,
                                       selector: Callable[[CorpusObject], bool] = lambda x: True,
                                       test_size: float = 0.2):
        """
        Evaluate the performance of predictive features (Classifier.pred_feats) in predicting for the label,
        using a train-test split.

        Run either on a Corpus (with Classifier labeller, selector, obj_type settings) or a list of Corpus objects

        :param corpus: target Corpus
        :param objs: target list of Corpus objects
        :param selector: if running on a Corpus, this is a (lambda) function that takes a Corpus object and returns True or False (i.e. include / exclude). By default, the selector includes all objects of the specified type in the Corpus.
        :param test_size: size of test set
        :return: accuracy and confusion matrix
        """
        X, y = extract_feats_and_label_bow(corpus, objs, self.obj_type, self.vector_name, self.labeller, selector)

        print("Running a train-test-split evaluation...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        self.clf.fit(X_train, y_train)
        preds = self.clf.predict(X_test)
        accuracy = np.mean(preds == y_test)
        print("Done.")
        return accuracy, confusion_matrix(y_true=y_test, y_pred=preds)

    def evaluate_with_cv(self, corpus: Corpus = None,
                         objs: List[CorpusObject] = None,
                         cv=KFold(n_splits=5),
                         selector: Callable[[CorpusObject], bool] = lambda x: True
                         ):
        """
        Evaluate the performance of predictive features (Classifier.pred_feats) in predicting for the label,
        using cross-validation for data splitting.

        Run either on a Corpus (with Classifier labeller, selector, obj_type settings) or a list of Corpus objects.

        :param corpus: target Corpus
        :param objs: target list of Corpus objects (do not pass in corpus if using this)
        :param cv: cross-validation model to use: KFold(n_splits=5) by default.
        :param selector: if running on a Corpus, this is a (lambda) function that takes a Corpus object and returns True or False (i.e. include / exclude). By default, the selector includes all objects of the specified type in the Corpus.

        :return: cross-validated accuracy score
        """

        X, y = extract_feats_and_label_bow(corpus, objs, self.obj_type, self.vector_name, self.labeller, selector)
        print("Running a cross-validated evaluation...")
        score = cross_val_score(self.clf, X, y, cv=cv)
        print("Done.")
        return score


