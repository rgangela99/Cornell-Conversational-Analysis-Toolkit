import unittest
from convokit.model import Utterance, Speaker, Corpus
from convokit import download


class CorpusLoadAndDump(unittest.TestCase):
    """
    Load a variety of existing (small) corpora to verify that there are no backward compatibility issues
    """

    def test_load_subreddit(self):
        corpus = Corpus(download('subreddit-hey'))
        corpus.dump('subreddit')

    def test_load_tennis(self):
        corpus = Corpus(download('tennis-corpus'))
        corpus.dump('tennis-corpus')

    def test_load_politeness(self):
        corpus = Corpus(download('wikipedia-politeness-corpus'))
        corpus.dump('wikipedia-politeness-corpus')

    def test_load_switchboard(self):
        corpus = Corpus(download("switchboard-corpus"))
        corpus.dump('switchboard-corpus')

if __name__ == '__main__':
    unittest.main()
