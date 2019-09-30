from convokit.transformer import Transformer
from convokit.model import Corpus
from datetime import datetime

class Datetimer(Transformer):
    def __init__(self, target="conversation"):
        if target not in ["conversation", "utterance"]:
            raise ValueError("Target must be 'conversation' or 'utterance'.")
        self.target = target

    def transform(self, corpus: Corpus):
        if self.target == "conversation":
            for convo in corpus.iter_conversations():
                if "timestamp" not in convo.meta:
                    print("Missing timestamp information for conversation id: {}".format(convo.id))
                else:
                    year, month, day, hour, min = datetime.utcfromtimestamp(convo.meta['timestamp'])
                    convo.meta["year"] = year
                    convo.meta["month"] = month
                    convo.meta["day"] = day
        elif self.target == "utterance":
            for utt in corpus.iter_utterances():
                if utt.timestamp is None:
                    print("Missing timestamp information for utterance id: {}".format(utt.id))
                else:
                    year, month, day, hour, min = datetime.utcfromtimestamp(utt.timestamp)
                    utt.meta["year"] = year
                    utt.meta["month"] = month
                    utt.meta["day"] = day
        return corpus