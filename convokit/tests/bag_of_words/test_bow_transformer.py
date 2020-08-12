from unittest import TestCase
import unittest

from numpy.testing import assert_array_equal
from scipy.sparse import csr_matrix

from convokit import BoWTransformer
from convokit.tests.util import burr_sir_corpus, BURR_SIR_TEXT_1, BURR_SIR_TEXT_2


def burr_sir_sentence_1_vector():
    indices = [0, 1, 2, 3, 5, 6, 7, 10]
    indptr = [0, 8]
    data = [1, 1, 1, 1, 1, 1, 1, 1]

    return csr_matrix((data, indices, indptr))


def burr_sir_sentence_2_vector():
    indices = [2, 4, 8, 9]
    indptr = [0, 4]
    data = [1, 1, 1, 1]

    return csr_matrix((data, indices, indptr))


def assert_sparse_matrices_equal(matrix1, matrix2):
    assert matrix1.shape == matrix2.shape
    assert (matrix1 != matrix2).nnz == 0


class FakeVectorizer:
    def transform(self, texts):
        if texts[0] == BURR_SIR_TEXT_1:
            return burr_sir_sentence_1_vector()
        elif texts[0] == BURR_SIR_TEXT_2:
            return burr_sir_sentence_2_vector()
        else:
            raise AssertionError

    def fit(self, docs):
        pass

# class TestBoWTransformer(TestCase):
#     def test_transform_utterances(self):
#         corpus = burr_sir_corpus()
#         corpus.print_summary_stats()
#         transformer = BoWTransformer(obj_type='utterance', vectorizer=FakeVectorizer())
#         corpus = transformer.fit_transform(corpus)
#
#         expected_vectors = [
#             burr_sir_sentence_1_vector(),
#             burr_sir_sentence_2_vector()
#         ]
#
#         for expected_vector, utterance in zip(expected_vectors, corpus.iter_utterances()):
#             actual_vector = utterance.meta['bow_vector']
#             assert_sparse_matrices_equal(expected_vector, actual_vector)


if __name__ == "__main__":
    unittest.main()
