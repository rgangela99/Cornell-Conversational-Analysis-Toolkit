from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, cross_val_score
from typing import List, Callable
from convokit import Transformer, CorpusObject, Corpus
from .util import *
from convokit.classifier.util import get_coefs_helper


class PairedPrediction(Transformer):
    def __init__(self, obj_type: str,
                 pred_feats: List[str],
                 selector: Callable[[CorpusObject], bool] = lambda x: True,
                 clf=None, # annotate_pairs: bool = True,
                 pair_id_feat_name: str = "pair_id",
                 label_feat_name: str = "label",
                 pair_orientation_feat_name: str = "pair_orientation"):

        """
        :param pred_feats: List of metadata features to be used in prediction. Features can either be values or a
                        dictionary of key-value pairs, but not a nested dictionary
        :param selector: optional function to filter object for
        :param clf: optional classifier to be used in the paired prediction
        :param pair_id_feat_name: metadata feature name to use in annotating object with pair id, default: "pair_id"
        :param label_feat_name: metadata feature name to use in annotating object with predicted label, default: "label"
        :param pair_orientation_feat_name: metadata feature name to use in annotating object with pair orientation,
        default: "pair_orientation"

        """
        # :param annotate_pairs: set to False if objects already have the pair information annotated

        assert obj_type in ["user", "utterance", "conversation"]
        self.obj_type = obj_type
        self.clf = Pipeline([("standardScaler", StandardScaler(with_mean=False)),
                             ("logreg", LogisticRegression(solver='liblinear'))]) if clf is None else clf
        self.pred_feats = pred_feats
        self.selector = selector
        self.pair_id_feat_name = pair_id_feat_name
        self.label_feat_name = label_feat_name
        self.pair_orientation_feat_name = pair_orientation_feat_name

    def fit(self, corpus: Corpus, y=None):
        # Check if Pairer.transform() needs to be run first
        self._check_for_pair_information(corpus)
        pair_id_to_objs = generate_pair_id_to_objs(corpus, self.obj_type, self.selector, self.pair_orientation_feat_name,
                                                   self.label_feat_name, self.pair_id_feat_name)

        X, y = generate_paired_X_y(self.pred_feats, self.pair_orientation_feat_name, pair_id_to_objs)
        self.clf.fit(X, y)
        return self

    def transform(self, corpus: Corpus) -> Corpus:
        return corpus

    def _check_for_pair_information(self, corpus):
        # Check if transform() needs to be run first
        sample_obj = next(corpus.iter_objs(self.obj_type))
        meta_keys = set(sample_obj.meta)
        required_keys = {self.pair_orientation_feat_name, self.pair_id_feat_name, self.label_feat_name}
        required_keys -= meta_keys
        if len(required_keys) > 0:
            raise ValueError("Some metadata features required for paired prediction are missing: {}. "
                             "You may need to run Pairer.transform() first.".format(required_keys))

    def summarize(self, corpus: Corpus, cv=LeaveOneOut()):
        """
        Run PairedPrediction on the corpus with cross-validation
        :param corpus: target Corpus (must be annotated with pair information using PairedPrediction.transform())
        :param cv: optional CV model: default is LOOCV
        :return: cross-validation accuracy score
        """
        pair_id_to_objs = generate_pair_id_to_objs(corpus, self.obj_type, self.selector, self.pair_orientation_feat_name,
                                                   self.label_feat_name, self.pair_id_feat_name)

        X, y = generate_paired_X_y(self.pred_feats, self.pair_orientation_feat_name, pair_id_to_objs)
        return np.mean(cross_val_score(self.clf, X, y, cv=cv, error_score='raise'))

    def get_coefs(self, feature_names: List[str], coef_func=None):
        """
        Get dataframe of classifier coefficients
        :param feature_names: list of feature names to get coefficients for
        :param coef_func: function for accessing the list of coefficients from the classifier model; by default,
                            assumes it is a pipeline with a logistic regression component
        :return: DataFrame of features and coefficients, indexed by feature names
        """
        return get_coefs_helper(self.clf, feature_names, coef_func)

    #
    # def print_extreme_coefs(self, feature_names: List[str], num_features: Optional[int] = None):
    #     """
    #     Must be run after summarize()
    #     Prints the extreme coefficients of the trained classifier model for visual inspection, assuming
    #     it is a pipeline with a logistic regression component
    #     :param feature_names: list of feature names to inspect
    #     :param num_features: optional number of extreme coefficients to print
    #     :return: None (prints features)
    #     """
    #     coefs = self.clf.named_steps['logreg'].coef_[0].tolist()
    #
    #     assert len(feature_names) == len(coefs)
    #
    #     feats_coefs = sorted(list(zip(feature_names, coefs)), key=lambda x: x[1], reverse=True)
    #
    #     if num_features is None:
    #         num_features = len(feature_names) // 4
    #
    #     print()
    #     print("TOP {} FEATURES".format(num_features))
    #     for ft, coef in feats_coefs[:num_features]:
    #         print("{}: {:.3f}".format(ft, coef))
    #     print()
    #     print("BOTTOM {} FEATURES".format(num_features))
    #     for ft, coef in feats_coefs[-num_features:]:
    #         print("{}: {:.3f}".format(ft, coef))
    #     print()
    #
    #
