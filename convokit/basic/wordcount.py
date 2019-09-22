from convokit.transformer import Transformer
from convokit.model import Corpus

class WordCount(Transformer):
	
	'''
		computes wordcount per utterance.
		:param use_tokenized: whether or not to use nltk-tokenized output (requires tokenizer to be run)
	'''
	def __init__(self, use_tokenized=True):
		self.ATTR_NAME = 'wordcount'
		self.use_tokenized = use_tokenized
	
	def transform(self, corpus: Corpus):
		for utterance in corpus.iter_utterances():
			if self.use_tokenized:
				utterance.add_meta('wordcount', len(utterance.meta['tokens'].split()))
			else:
				utterance.add_meta('wordcount', len(utterance.text.split()))
		return corpus