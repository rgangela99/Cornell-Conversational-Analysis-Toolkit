import unittest

from convokit.model import Corpus, Speaker, Utterance
from convokit.text_processing.textParser import _process_sentence, _process_token, TextParser
from convokit.tests.util import buffalo_doc, fox_doc, fox_buffalo_doc, burr_sir_corpus, \
    BUFFALO_TEXT, FOX_TEXT, FOX_BUFFALO_TEXT, BURR_SIR_TEXT_1, BURR_SIR_TEXT_2, \
    burr_sir_doc_1, burr_sir_doc_2, BURR_SIR_SENTENCE_1, BURR_SIR_SENTENCE_2, BURR_SIR_SENTENCE_3, \
    BURR_SIR_SENTENCE_4, burr_sir_sentence_doc_1, burr_sir_sentence_doc_2, burr_sir_sentence_doc_3, \
    burr_sir_sentence_doc_4


class TestTextParser(unittest.TestCase):
    def test_process_token_tokenize_mode(self):
        doc = fox_doc()
        token = doc[3]

        expected = {
            'tok': 'fox'
        }
        actual = _process_token(token, mode='tokenize')
        self.assertDictEqual(expected, actual)


    def test_process_token_tag_mode(self):
        doc = fox_doc()
        token = doc[3]

        expected = {
            'tok': 'fox',
            'tag': 'NN'
        }
        actual = _process_token(token, mode='tag')
        self.assertDictEqual(expected, actual)

    def test_process_token_parse_mode(self):
        doc = fox_doc()
        token = doc[3]

        expected = {
            'tok': 'fox',
            'tag': 'NN',
            'dn': [],
            'up': 2,
            'dep': 'NN'
        }
        actual = _process_token(token, mode='parse', offset=2)
        self.assertDictEqual(expected, actual)

    def test_process_sentence_parse_mode(self):
        doc = fox_doc()

        expected = {
            'rt': 1,
            'toks': [
                {'tok': 'A', 'tag': 'DT', 'dep': 'det', 'up': 1, 'dn': []},
                {'tok': 'quick', 'tag': 'JJ', 'dep': 'amod', 'up': 1, 'dn': []},
                {'tok': 'brown', 'tag': 'JJ', 'dep': 'amod', 'up': 1, 'dn': []},
                {'tok': 'fox', 'tag': 'NN', 'dep': 'NN', 'up': 1, 'dn': []},
                {'tok': 'jumps', 'tag': 'NNS', 'dep': 'ROOT', 'dn': [-3, -2, -1, 0, 2]},
                {'tok': 'over', 'tag': 'IN', 'dep': 'prep', 'up': 1, 'dn': [5]},
                {'tok': 'the', 'tag': 'DT', 'dep': 'det', 'up': 5, 'dn': []},
                {'tok': 'lazy', 'tag': 'JJ', 'dep': 'amod', 'up': 5, 'dn': []},
                {'tok': 'dog', 'tag': 'NN', 'dep': 'pobj', 'up': 2, 'dn': [3, 4]}
            ]
        }
        actual = _process_sentence(next(doc.sents), mode='parse', offset=3)

        self.assertDictEqual(expected, actual)

    def test_process_sentence_nonparse_mode(self):
        doc = fox_doc()

        expected = {
            'toks': [
                {'tok': 'A'},
                {'tok': 'quick'},
                {'tok': 'brown'},
                {'tok': 'fox'},
                {'tok': 'jumps'},
                {'tok': 'over'},
                {'tok': 'the'},
                {'tok': 'lazy'},
                {'tok': 'dog'}
            ]
        }
        actual = _process_sentence(next(doc.sents), mode='tokenize', offset=3)

        self.assertDictEqual(expected, actual)

    def test_process_text_parse_mode(self):
        def fake_spacy_nlp(input_text):
            text_to_doc = {
                BURR_SIR_TEXT_1: burr_sir_doc_1(),
                BURR_SIR_TEXT_2: burr_sir_doc_2()
            }

            return text_to_doc[input_text]
        
        parser = TextParser(spacy_nlp=fake_spacy_nlp, mode='parse')
        corpus = burr_sir_corpus()
        actual = [utterance.meta['parsed'] for utterance in parser.transform(corpus).iter_utterances()]
        expected = [
            [
                {
                    'rt': 0,
                    'toks': [
                        {
                            'tok': 'Pardon',
                            'tag': 'VB',
                            'dep': 'ROOT',
                            'dn': [1, 2]
                        },
                        {
                            'tok': 'me',
                            'tag': 'PRP',
                            'dep': 'dobj',
                            'up': 0,
                            'dn': []
                        },
                        {
                            'tok': '.',
                            'tag': '.',
                            'dep': 'punct',
                            'up': 0,
                            'dn': []
                        }
                    ]
                },
                {
                    'rt': 0,
                    'toks': [
                        {
                            'tok': 'Are',
                            'tag': 'VBP',
                            'dep': 'ROOT',
                            'dn': [1, 3, 4, 5, 6]
                        },
                        {
                            'tok': 'you',
                            'tag': 'PRP',
                            'dep': 'nsubj',
                            'up': 0,
                            'dn': []
                        },
                        {
                            'tok': 'Aaron',
                            'tag': 'NNP',
                            'dep': 'compound',
                            'up': 3,
                            'dn': []
                        },
                        {
                            'tok': 'Burr',
                            'tag': 'NNP',
                            'dep': 'attr',
                            'up': 0,
                            'dn': [
                                2
                            ]
                        },
                        {
                            'tok': ',',
                            'tag': ',',
                            'dep': 'punct',
                            'up': 0,
                            'dn': []
                        },
                        {
                            'tok': 'sir',
                            'tag': 'NN',
                            'dep': 'npadvmod',
                            'up': 0,
                            'dn': []
                        },
                        {
                            'tok': '?',
                            'tag': '.',
                            'dep': 'punct',
                            'up': 0,
                            'dn': []
                        }
                    ]
                }
            ],
            [
                {
                    'rt': 1,
                    'toks': [
                        {
                            'tok': 'That',
                            'tag': 'DT',
                            'dep': 'nsubj',
                            'up': 1,
                            'dn': []
                        },
                        {
                            'tok': 'depends',
                            'tag': 'VBZ',
                            'dep': 'ROOT',
                            'dn': [
                                0,
                                2
                            ]
                        },
                        {
                            'tok': '.',
                            'tag': '.',
                            'dep': 'punct',
                            'up': 1,
                            'dn': []
                        }
                    ]
                },
                {
                    'rt': 2,
                    'toks': [
                        {
                            'tok': 'Who',
                            'tag': 'WP',
                            'dep': 'nsubj',
                            'up': 2,
                            'dn': []
                        },
                        {
                            'tok': "'s",
                            'tag': 'VBZ',
                            'dep': 'aux',
                            'up': 2,
                            'dn': []
                        },
                        {
                            'tok': 'asking',
                            'tag': 'VBG',
                            'dep': 'ROOT',
                            'dn': [0, 1, 3]
                        },
                        {
                            'tok': '?',
                            'tag': '.',
                            'dep': 'punct',
                            'up': 2,
                            'dn': []
                        }
                    ]
                }
            ]
        ]

        self.assertListEqual(expected, actual)

    def test_process_text_tag_mode(self):
        class FakeSentenceTokenizer:
            def tokenize(self, input_text):
                text_to_sentences = {
                    BURR_SIR_TEXT_1: [
                        'Pardon me.',
                        'Are you Aaron Burr, sir?'
                    ],
                    BURR_SIR_TEXT_2: [
                        'That depends.',
                        "Who's asking?"
                    ]
                }

                return text_to_sentences[input_text]

        def fake_spacy_nlp(input_text):
            text_to_doc = {
                BURR_SIR_SENTENCE_1: burr_sir_sentence_doc_1(),
                BURR_SIR_SENTENCE_2: burr_sir_sentence_doc_2(),
                BURR_SIR_SENTENCE_3: burr_sir_sentence_doc_3(),
                BURR_SIR_SENTENCE_4: burr_sir_sentence_doc_4()
            }

            return text_to_doc[input_text]
        
        parser = TextParser(spacy_nlp=fake_spacy_nlp, sent_tokenizer=FakeSentenceTokenizer(), mode='tag')
        corpus = burr_sir_corpus()
        actual = [utterance.meta['parsed'] for utterance in parser.transform(corpus).iter_utterances()]
        expected = [
            [
                {
                    "toks": [
                        {
                            "tok": "Pardon",
                            "tag": "VB"
                        },
                        {
                            "tok": "me",
                            "tag": "PRP"
                        },
                        {
                            "tok": ".",
                            "tag": "."
                        }
                    ]
                },
                {
                    "toks": [
                        {
                            "tok": "Are",
                            "tag": "VBP"
                        },
                        {
                            "tok": "you",
                            "tag": "PRP"
                        },
                        {
                            "tok": "Aaron",
                            "tag": "NNP"
                        },
                        {
                            "tok": "Burr",
                            "tag": "NNP"
                        },
                        {
                            "tok": ",",
                            "tag": ","
                        },
                        {
                            "tok": "sir",
                            "tag": "NN"
                        },
                        {
                            "tok": "?",
                            "tag": "."
                        }
                    ]
                }
            ],
            [
                {
                    "toks": [
                        {
                            "tok": "That",
                            "tag": "DT"
                        },
                        {
                            "tok": "depends",
                            "tag": "VBZ"
                        },
                        {
                            "tok": ".",
                            "tag": "."
                        }
                    ]
                },
                {
                    "toks": [
                        {
                            "tok": "Who",
                            "tag": "WP"
                        },
                        {
                            "tok": "'s",
                            "tag": "VBZ"
                        },
                        {
                            "tok": "asking",
                            "tag": "VBG"
                        },
                        {
                            "tok": "?",
                            "tag": "."
                        }
                    ]
                }
            ]
        ]

        self.assertListEqual(expected, actual)
    
    def test_process_text_tokenize_mode(self):
        class FakeSentenceTokenizer:
            def tokenize(self, input_text):
                text_to_sentences = {
                    BURR_SIR_TEXT_1: [
                        'Pardon me.',
                        'Are you Aaron Burr, sir?'
                    ],
                    BURR_SIR_TEXT_2: [
                        'That depends.',
                        "Who's asking?"
                    ]
                }

                return text_to_sentences[input_text]


        def fake_spacy_nlp(input_text):
            text_to_doc = {
                BURR_SIR_SENTENCE_1: burr_sir_sentence_doc_1(),
                BURR_SIR_SENTENCE_2: burr_sir_sentence_doc_2(),
                BURR_SIR_SENTENCE_3: burr_sir_sentence_doc_3(),
                BURR_SIR_SENTENCE_4: burr_sir_sentence_doc_4()
            }

            return text_to_doc[input_text]
        
        parser = TextParser(spacy_nlp=fake_spacy_nlp, sent_tokenizer=FakeSentenceTokenizer(), mode='tokenize')
        corpus = burr_sir_corpus()
        actual = [utterance.meta['parsed'] for utterance in parser.transform(corpus).iter_utterances()]
        expected = [
            [
                {
                    "toks": [
                        {
                            "tok": "Pardon"
                        },
                        {
                            "tok": "me"
                        },
                        {
                            "tok": "."
                        }
                    ]
                },
                {
                    "toks": [
                        {
                            "tok": "Are"
                        },
                        {
                            "tok": "you"
                        },
                        {
                            "tok": "Aaron"
                        },
                        {
                            "tok": "Burr"
                        },
                        {
                            "tok": ","
                        },
                        {
                            "tok": "sir"
                        },
                        {
                            "tok": "?"
                        }
                    ]
                }
            ],
            [
                {
                    "toks": [
                        {
                            "tok": "That"
                        },
                        {
                            "tok": "depends"
                        },
                        {
                            "tok": "."
                        }
                    ]
                },
                {
                    "toks": [
                        {
                            "tok": "Who"
                        },
                        {
                            "tok": "'s"
                        },
                        {
                            "tok": "asking"
                        },
                        {
                            "tok": "?"
                        }
                    ]
                }
            ]
        ]

        self.assertListEqual(expected, actual)


if __name__ == '__main__':
    unittest.main()
