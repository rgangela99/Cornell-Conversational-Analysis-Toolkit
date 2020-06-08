from copy import deepcopy
import unittest

from convokit.tests.util import burr_sir_corpus
from convokit.text_processing.textCleaner import clean_str, TextCleaner


class TestTextCleaner(unittest.TestCase):
    def test_no_options(self):
        corpus = burr_sir_corpus()
        cleaner = TextCleaner(text_cleaner=lambda text: 'cleaned text')
        cleaned_corpus = cleaner.transform(deepcopy(corpus))

        for utterance in cleaned_corpus.iter_utterances():
            self.assertEqual(utterance.text, 'cleaned text')

    def test_dont_replace_text(self):
        corpus = burr_sir_corpus()
        cleaner = TextCleaner(text_cleaner=lambda text: 'cleaned text', replace_text=False)
        cleaned_corpus = cleaner.transform(deepcopy(corpus))

        for original_utterance, cleaned_utterance in zip(corpus.iter_utterances(), cleaned_corpus.iter_utterances()):
            self.assertEqual(original_utterance.text, cleaned_utterance.text)
            self.assertEqual(cleaned_utterance.meta['cleaned'], 'cleaned text')


    def test_save_original(self):
        corpus = burr_sir_corpus()
        cleaner = TextCleaner(text_cleaner=lambda text: 'cleaned text', replace_text=True, save_original=True)
        cleaned_corpus = cleaner.transform(deepcopy(corpus))

        for original_utterance, cleaned_utterance in zip(corpus.iter_utterances(), cleaned_corpus.iter_utterances()):
            self.assertEqual(cleaned_utterance.text, 'cleaned text')
            self.assertEqual(original_utterance.text, cleaned_utterance.meta['original'])
    
    def test_clean_str_replacements(self):
        original_str = 'https://mywebsite.com myemail@gmail.com (123) 456-7890 1,000 $'
        cleaned_str = clean_str(original_str)

        self.assertEqual('<url> <email> <phone> <number> <cur>', cleaned_str)
    
    def test_clean_str_text(self):
        original_str = 'BA OBAMA ĐỘI BA Ô BA OBAMA' # vietnamese joke https://twitter.com/tetracarbon/status/735009694515789824
        cleaned_str = clean_str(original_str)

        self.assertEqual('ba obama doi ba o ba obama', cleaned_str)
