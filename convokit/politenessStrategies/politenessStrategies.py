from typing import Callable, Optional
from convokit.model import Utterance
from convokit.politeness_api.features.politeness_strategies import get_politeness_strategy_features
from convokit.transformer import Transformer
from convokit.model import Corpus
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class PolitenessStrategies(Transformer):
    """
    Encapsulates extraction of politeness strategies from utterances in a
    Corpus.

    :param verbose: whether or not to print status messages while computing features
    """

    def __init__(self, verbose: bool=False):
        self.ATTR_NAME = "politeness_strategies"
        self.MRKR_NAME = "politeness_markers"
        self.verbose = verbose

    def transform(self, corpus: Corpus, selector: Optional[Callable[[Utterance], bool]] = lambda utt: True,
                  markers: bool = False):
        """
        Extract politeness strategies from each utterances in the corpus and annotate
        the utterances with the extracted strategies. Requires that the corpus has previously
        been transformed by a Parser, such that each utterance has dependency parse info in
        its metadata table.

        :param corpus: the corpus to compute features for.
        :param selector: a (lambda) function that takes an Utterance and returns a bool indicating whether the utterance should be included in this annotation step.
        :param markers: whether or not to add politeness occurrence markers
        """
        for utt in corpus.iter_utterances():
            if selector(utt):
                for i, sent in enumerate(utt.meta["parsed"]):
                    for p in sent["toks"]:
                        p["tok"] = re.sub("[^a-z,.:;]", "", p["tok"].lower())
                utt.meta[self.ATTR_NAME], marks = get_politeness_strategy_features(utt)

                if markers:
                    utt.meta[self.MRKR_NAME] = marks
            else:
                utt.meta[self.ATTR_NAME] = None
                utt.meta[self.MRKR_NAME] = None

        return corpus

    def _get_scores(self, corpus: Corpus, selector: Optional[Callable[[Utterance], bool]] = lambda utt: True):
        """
        Calculates average occurrence per utterance. Used in summarize()

        :param corpus: the target Corpus
        :param selector: (lambda) function specifying whether the utterance should be included
        """

        utts = list(corpus.iter_utterances(selector))
        if self.MRKR_NAME not in utts[0].meta:
            print("Could not find politeness markers metadata. Running transform() on corpus first...", end="")
            self.transform(corpus, markers=True)
            print("Done.")

        counts = {k[21:len(k)-2]: 0 for k in utts[0].meta[self.MRKR_NAME].keys()}

        for utt in utts:
            for k, v in utt.meta[self.MRKR_NAME].items():
                counts[k[21: len(k)-2]] += len(v)
        scores = {k: v/len(utts) for k, v in counts.items()}
        return scores

    def summarize(self, corpus: Corpus, selector: Callable[[Utterance], bool] = lambda utt: True, plot: bool = False, y_lim = None):
        """
        Calculates average occurrence per utterance and outputs graph if plot == True, with an optional selector
        that specifies which utterances to include in the analysis

        :param corpus: the target Corpus
        :param selector: a function (typically, a lambda function) that takes an Utterance and returns True or False (i.e. include / exclude).
		By default, the selector includes all Utterances in the Corpus.
        :param plot: whether or not to output graph.
        :return: a pandas DataFrame of scores with graph optionally outputted
        """
        scores = self._get_scores(corpus, selector)

        if plot:
            plt.figure(dpi=200, figsize=(9, 6))
            plt.bar(list(range(len(scores))), scores.values(), tick_label = list(scores.keys()), align="edge")
            plt.xticks(np.arange(.4, len(scores)+.4), rotation=45, ha="right")
            plt.ylabel("Occurrences per Utterance", size=20)
            plt.yticks(size=15)
            if y_lim != None:
                plt.ylim(0, y_lim)
            plt.show()

        return pd.DataFrame.from_dict(scores, orient='index', columns=["Averages"])
