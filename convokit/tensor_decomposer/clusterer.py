from convokit import Transformer, Corpus
from sklearn import metrics
from sklearn.cluster import KMeans
from statistics import mode
import numpy as np


class Clusterer(Transformer):

    def fit(self, corpus: Corpus, y=None):
        pass

    def transform(self, corpus: Corpus, **kwargs) -> Corpus:
        pass


    @staticmethod
    def purity(matrix, y_true, n_clusters=3):
        kmeans = KMeans(n_clusters=n_clusters, random_state=2020).fit(matrix)
        y_pred = kmeans.predict(matrix)

        y_pred_cluster0 = y_true[y_pred == 0]
        y_pred_cluster1 = y_true[y_pred == 1]
        y_pred_cluster2 = y_true[y_pred == 2]

        correct = np.sum(y_pred_cluster0 == mode(y_pred_cluster0)) + \
                  np.sum(y_pred_cluster1 == mode(y_pred_cluster1)) + \
                  np.sum(y_pred_cluster2 == mode(y_pred_cluster2))
        return correct / len(y_pred)

    def summarize(self, corpus: Corpus, **kwargs):
        pass