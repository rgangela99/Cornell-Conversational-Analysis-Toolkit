import unittest

from convokit.tests.util import parsed_burr_sir_corpus
from convokit.phrasing_motifs.questionSentences import QuestionSentences


def parsed_burr_sir_corpus_with_lowercase_are():
    corpus = parsed_burr_sir_corpus()
    for utterance in corpus.iter_utterances():
        parsed = utterance.get_info('parsed')
        for sentence in parsed:
            if sentence['toks'][0]['tok'] == 'Are':
                sentence['toks'][0]['tok'] = 'are'
    
    return corpus


class TestQuestionSentences(unittest.TestCase):
    def test_use_caps(self):
        corpus = parsed_burr_sir_corpus_with_lowercase_are()
        transformer = QuestionSentences(input_field='sentences', output_field='questions', use_caps=True)
        transformed_corpus = transformer.transform(corpus)

        expected_sentences_list = [
            [],
            ["Who's asking?"]
        ]
        for utterance, expected_sentences in zip(transformed_corpus.iter_utterances(), expected_sentences_list):
            self.assertListEqual(expected_sentences, utterance.get_info('questions'))

    def test_dont_use_caps(self):
        corpus = parsed_burr_sir_corpus_with_lowercase_are()
        transformer = QuestionSentences(input_field='sentences', output_field='questions', use_caps=False)
        transformed_corpus = transformer.transform(corpus)

        expected_sentences_list = [
            ['Are you Aaron Burr, sir?'],
            ["Who's asking?"]
        ]
        for utterance, expected_sentences in zip(transformed_corpus.iter_utterances(), expected_sentences_list):
            self.assertListEqual(expected_sentences, utterance.get_info('questions'))
