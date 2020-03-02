from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, cross_val_score
from typing import List, Callable
from convokit import CorpusObject, Corpus
from .util import *
from .pairedPrediction import PairedPrediction
from convokit.classifier.util import get_coefs_helper


class PairedBoW(PairedPrediction):
    """
    Transformer for doing a Paired Prediction with bag-of-words vectors.

    :param pairing_func: the Corpus object characteristic to pair on, e.g. to pair on the first 10 characters of a well structured id, use lambda obj: obj.id[:10]
    :param pos_label_func: The function to check if the object is a positive instance
    :param neg_label_func: The function to check if the object is a negative instance
    :param clf: optional classifier to be used in the paired prediction
    :param pair_id_feat_name: metadata feature name to use in annotating object with pair id, default: "pair_id"
    :param label_feat_name: metadata feature name to use in annotating object with predicted label, default: "label"
    :param pair_orientation_feat_name: metadata feature name to use in annotating object with pair orientation, default: "pair_orientation"

    """
    def __init__(self, obj_type: str,
                 vector_name="bow_vector",
                 clf=None, pair_id_feat_name: str = "pair_id",
                 label_feat_name: str = "pair_obj_label",
                 pair_orientation_feat_name: str = "pair_orientation"):

        assert obj_type in ["user", "utterance", "conversation"]
        self.obj_type = obj_type
        self.vector_name = vector_name

        clf = Pipeline([("standardScaler", StandardScaler(with_mean=False)),
                        ("logreg", LogisticRegression(solver='liblinear'))]) if clf is None else clf

        super().__init__(obj_type=obj_type, pred_feats=[],
                         pair_id_feat_name=pair_id_feat_name,
                         label_feat_name=label_feat_name, pair_orientation_feat_name=pair_orientation_feat_name, clf=clf)

    def fit(self, corpus: Corpus, y=None, selector: Callable[[CorpusObject], bool] = lambda x: True):
        # Check if Pairer.transform() needs to be run first
        self._check_for_pair_information(corpus)
        pair_id_to_objs = generate_pair_id_to_objs(corpus, self.obj_type, selector, self.pair_orientation_feat_name,
                                                   self.label_feat_name, self.pair_id_feat_name)

        X, y = generate_bow_paired_X_y(self.pair_orientation_feat_name, pair_id_to_objs, self.vector_name)
        self.clf.fit(X, y)
        return self

    def summarize(self, corpus: Corpus, selector: Callable[[CorpusObject], bool] = lambda x: True, cv=LeaveOneOut()):
        """
        Run PairedPrediction on the corpus with cross-validation

        :param corpus: annoted Corpus (with pair information from PairedPrediction.transform())
        :param selector: selector (lambda) function for which objects should be included in the analysis
        :param cv: optional CV model: default is LOOCV
        :return: cross-validation accuracy score
        """
        pair_id_to_objs = generate_pair_id_to_objs(corpus, self.obj_type, selector, self.pair_orientation_feat_name,
                                                   self.label_feat_name, self.pair_id_feat_name)

        X, y = generate_bow_paired_X_y(self.pair_orientation_feat_name, pair_id_to_objs, self.vector_name)
        return np.mean(cross_val_score(self.clf, X, y, cv=cv, error_score='raise'))


    def get_coefs(self, feature_names: List[str], coef_func=None):
        """
        Get dataframe of classifier coefficients. By default, assumes it is a pipeline with a logistic regression component.

        :param feature_names: list of feature names to get coefficients for. if None, uses vectorizer vocabulary
        :param coef_func: function for accessing the list of coefficients from the classifier model
        :return: DataFrame of features and coefficients, indexed by feature names
        """
        return get_coefs_helper(self.clf, feature_names, coef_func)

