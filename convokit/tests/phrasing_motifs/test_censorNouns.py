import unittest

from convokit.tests.util import parsed_burr_sir_corpus
from convokit.phrasing_motifs.censorNouns import CensorNouns

class TestCensorNouns(unittest.TestCase):
    def test_censor_nouns(self):
        transformer = CensorNouns(output_field='censored')
        corpus = parsed_burr_sir_corpus()

        transformed_corpus = transformer.transform(corpus)
        expected_censored_list = [
            [
                {
                    "rt": 0,
                    "toks": [
                        {
                            "dep": "ROOT",
                            "dn": [
                                1,
                                2
                            ],
                            "tag": "VB",
                            "tok": "pardon"
                        },
                        {
                            "dep": "dobj",
                            "dn": [],
                            "tag": "PRP",
                            "up": 0,
                            "tok": "NN~"
                        },
                        {
                            "dep": "punct",
                            "dn": [],
                            "tag": ".",
                            "up": 0,
                            "tok": "."
                        }
                    ]
                },
                {
                    "rt": 0,
                    "toks": [
                        {
                            "dep": "ROOT",
                            "dn": [
                                1,
                                3,
                                4,
                                5,
                                6
                            ],
                            "tag": "VBP",
                            "tok": "are"
                        },
                        {
                            "dep": "nsubj",
                            "dn": [],
                            "tag": "PRP",
                            "up": 0,
                            "tok": "NN~"
                        },
                        {
                            "dep": "compound",
                            "dn": [],
                            "tag": "NNP",
                            "up": 3,
                            "tok": "NN~"
                        },
                        {
                            "dep": "attr",
                            "dn": [
                                2
                            ],
                            "tag": "NNP",
                            "up": 0,
                            "tok": "NN~"
                        },
                        {
                            "dep": "punct",
                            "dn": [],
                            "tag": ",",
                            "up": 0,
                            "tok": ","
                        },
                        {
                            "dep": "npadvmod",
                            "dn": [],
                            "tag": "NN",
                            "up": 0,
                            "tok": "NN~"
                        },
                        {
                            "dep": "punct",
                            "dn": [],
                            "tag": ".",
                            "up": 0,
                            "tok": "?"
                        }
                    ]
                }
            ],
            [
                {
                    "rt": 1,
                    "toks": [
                        {
                            "dep": "nsubj",
                            "dn": [],
                            "tag": "DT",
                            "up": 1,
                            "tok": "NN~"
                        },
                        {
                            "dep": "ROOT",
                            "dn": [
                                0,
                                2
                            ],
                            "tag": "VBZ",
                            "tok": "depends"
                        },
                        {
                            "dep": "punct",
                            "dn": [],
                            "tag": ".",
                            "up": 1,
                            "tok": "."
                        }
                    ]
                },
                {
                    "rt": 2,
                    "toks": [
                        {
                            "dep": "nsubj",
                            "dn": [],
                            "tag": "WP",
                            "up": 2,
                            "tok": "who"
                        },
                        {
                            "dep": "aux",
                            "dn": [],
                            "tag": "VBZ",
                            "up": 2,
                            "tok": "'s"
                        },
                        {
                            "dep": "ROOT",
                            "dn": [
                                0,
                                1,
                                3
                            ],
                            "tag": "VBG",
                            "tok": "asking"
                        },
                        {
                            "dep": "punct",
                            "dn": [],
                            "tag": ".",
                            "up": 2,
                            "tok": "?"
                        }
                    ]
                }
            ]
        ]
        for expected_censored, utterance in zip(expected_censored_list, transformed_corpus.iter_utterances()):
            self.assertListEqual(expected_censored, utterance.retrieve_meta('censored'))
