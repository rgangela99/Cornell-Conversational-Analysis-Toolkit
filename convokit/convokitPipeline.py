from sklearn.pipeline import Pipeline
from convokit.model import Utterance

class ConvokitPipeline(Pipeline):
	"""
		A pipeline of transformers. Builds on and inherits functionality from scikit-learn's Pipeline class.

		:param steps: a list of (name, transformer) tuples in the order that they are to be called.
	"""

	def __init__(self, steps):
		Pipeline.__init__(self, steps)

	def transform_utterance(self, utt):
		"""
			Computes attributes of an individual string or utterance using all of the transformers in the pipeline.
			
			:param utt: the utterance to compute attributes for.
			:return: the utterance, with new attributes.
		"""

		if isinstance(utt, str):
			utt = Utterance(text=utt)
		for name, transform in self.steps:
			utt = transform.transform_utterance(utt)
		return utt