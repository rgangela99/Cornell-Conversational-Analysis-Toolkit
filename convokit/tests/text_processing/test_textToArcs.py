import unittest

from convokit.tests.util import parsed_burr_sir_corpus
from convokit.text_processing.textToArcs import TextToArcs

class TestTextToArcs(unittest.TestCase):
    def test_default_options(self):
        transformer = TextToArcs(output_field='arcs')
        corpus = parsed_burr_sir_corpus()

        transformed_corpus = transformer.transform(corpus)
        expected_arcs = [
            [
                'me_* pardon>* pardon_* pardon_me',
                'aaron_* are>* are_* are_burr are_sir are_you burr_* burr_aaron sir_* you_*'
            ],
            [
                'depends_* depends_that that>* that_*',
                "'s_* asking_'s asking_* asking_who who>'s who>* who_*"
            ]
        ]
        for expected_arc, utterance in zip(expected_arcs, transformed_corpus.iter_utterances()):
            self.assertListEqual(expected_arc, utterance.meta['arcs'])

    def test_dont_use_start(self):
        transformer = TextToArcs(output_field='arcs', use_start=False)
        corpus = parsed_burr_sir_corpus()

        transformed_corpus = transformer.transform(corpus)
        expected_arcs = [
            [
                'me_* pardon_* pardon_me',
                'aaron_* are_* are_burr are_sir are_you burr_* burr_aaron sir_* you_*'
            ],
            [
                'depends_* depends_that that_*',
                "'s_* asking_'s asking_* asking_who who_*"
            ]
        ]
        for expected_arc, utterance in zip(expected_arcs, transformed_corpus.iter_utterances()):
            self.assertListEqual(expected_arc, utterance.meta['arcs'])

    def test_root_only(self):
        transformer = TextToArcs(output_field='arcs', root_only=True)
        corpus = parsed_burr_sir_corpus()

        transformed_corpus = transformer.transform(corpus)
        expected_arcs = [
            [
                'pardon>* pardon_* pardon_me',
                'are>* are_* are_burr are_sir are_you'
            ],
            [
                'depends_* depends_that that>*',
                "asking_'s asking_* asking_who who>'s who>*"
            ]
        ]
        for expected_arc, utterance in zip(expected_arcs, transformed_corpus.iter_utterances()):
            self.assertListEqual(expected_arc, utterance.meta['arcs'])
