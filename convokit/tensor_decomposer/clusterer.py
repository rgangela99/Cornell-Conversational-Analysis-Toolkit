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
    def purity(matrix, n_clusters, actual_num_clusters=3, group_size=333):
        kmeans = KMeans(n_clusters=n_clusters, random_state=2020).fit(matrix)
        y_pred = kmeans.predict(matrix)

        y_true = np.zeros(actual_num_clusters * group_size)
        for i in range(actual_num_clusters):
            y_true[i*group_size:(i+1)*group_size] = i

        correct = sum(np.sum(y_true[y_pred==i] == mode(y_true[y_pred==i])) for i in range(n_clusters))

        return correct / len(y_pred)

    def summarize(self, corpus: Corpus, **kwargs):
        pass