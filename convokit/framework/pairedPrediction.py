from .framework import Framework
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.model_selection import train_test_split
from typing import List, Union
from pandas import DataFrame


class PairedPrediction(Framework):
    def __init__(self, exclude_na=True):
        self.clf = Pipeline([("standardScaler", StandardScaler()), ("logreg", LogisticRegression(solver='liblinear'))])
        self.exclude_na = exclude_na

    def fit(self, X_train, y_train):
        print("Train accuracy: {:.4f}".format(self.clf.fit(X_train, y_train).score(X_train, y_train)))
        return self

    def predict(self, X_test, y_test):
        test_acc = self.clf.score(X_test, y_test)
        print("Test accuracy: {:.4f}".format(test_acc))
        return test_acc

    def print_extreme_coefs(self, feature_names: List[str], num_features: int = 5):
        coefs = self.clf.named_steps['logreg'].coef_[0].tolist()

        assert len(feature_names) == len(coefs)

        feats_coefs = sorted(list(zip(feature_names, coefs)), key=lambda x: x[1], reverse=True)

        print()
        print("TOP {} FEATURES".format(num_features))
        for ft, coef in feats_coefs[:num_features]:
            print("{}: {:.3f}".format(ft, coef))
        print()
        print("BOTTOM {} FEATURES".format(num_features))
        for ft, coef in feats_coefs[-num_features:]:
            print("{}: {:.3f}".format(ft, coef))
        print()

    def _generate_paired_X_y(self, feats: DataFrame, pos_ids: List[str], neg_ids: List[str]):
        """

        :param feats: Features dataframe indexed by ids
        :param pos_ids: Index ids that correspond to positive class
        :param neg_ids: Index ids that correspond to negative class
        :return:
        """
        X, y = [], []
        flip = True

        excluded = 0
        for idx in range(len(pos_ids)):
            pos_feats = np.array(feats.loc[pos_ids[idx]])
            neg_feats = np.array(feats.loc[neg_ids[idx]])

            if self.exclude_na and (np.isnan(pos_feats).any() or np.isnan(neg_feats).any()):
                excluded += 1
                continue

            if flip:
                y.append(1)
                diff = pos_feats - neg_feats
            else:
                y.append(0)
                diff = neg_feats - pos_feats

            X.append(diff)
            flip = not flip

        if excluded > 0:
            print("Excluded {} data point(s) that contained NaN values.".format(excluded))

        return np.array(X), np.array(y)

    def fit_predict(self, feats: DataFrame, pos_ids: List[str], neg_ids: List[str], test_size=0.2):
        """

        :param feats: Features dataframe indexed by ids
        :param pos_ids: Index ids that correspond to positive class (paired with neg_id at same list idnex)
        :param neg_ids: Index ids that correspond to negative class (paired with pos_id at same list index)
        :return:
        """
        assert len(pos_ids) == len(neg_ids)
        X, y = self._generate_paired_X_y(feats, pos_ids, neg_ids)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        self.fit(X_train, y_train)
        self.predict(X_test, y_test)



