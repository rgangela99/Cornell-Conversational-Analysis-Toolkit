from typing import Callable, Optional
from convokit.model import Utterance
from convokit.politeness_api.features.politeness_strategies import get_politeness_strategy_features
from convokit.text_processing import process_text
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
    
    :param strategy_attribute_name: metadata attribute name to store politeness strategies features under during the `transform()` step.  Default is 'politeness_strategies'. 
    :param marker_attribute_name: metadata attribute name to store politeness markers under during the `transform()` step. Default is 'politeness_markers'.
    :param verbose: whether or not to print status messages while computing features.
    """

    def __init__(self, strategy_attribute_name="politeness_strategies", marker_attribute_name="politeness_markers", verbose: bool=False):
        self.strategy_attribute_name = strategy_attribute_name
        self.marker_attribute_name = marker_attribute_name
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
                utt.meta[self.strategy_attribute_name], marks = get_politeness_strategy_features(utt)

                if markers:
                    utt.meta[self.marker_attribute_name] = marks
            else:
                utt.meta[self.strategy_attribute_name] = None
                utt.meta[self.marker_attribute_name] = None

        return corpus
    
    
    def transform_utterance(self, utterance, spacy_nlp = None, markers = False):
        
        """
        Extract politeness strategies for raw string inputs. 
        
        :param utterance: the utterance to be annotated with politeness strategies. 
        :spacy_nlp: if provided, will use this SpaCy object to do parsing; otherwise will initialize an object via `load('en')`.
        :return: the utterance with politeness annotations.
        """
        
        if isinstance(utterance, str):
            utterance = Utterance(text=utterance)
        
        if spacy_nlp is None:
            spacy_nlp = spacy.load('en', disable=['ner'])
            
        utterance.meta['parsed'] = process_text(utterance.text, spacy_nlp=spacy_nlp)
        
        for i, sent in enumerate(utterance.meta["parsed"]):
            for p in sent["toks"]:
                p["tok"] = re.sub("[^a-z,.:;]", "", p["tok"].lower())
            
        utterance.meta[self.strategy_attribute_name], marks = get_politeness_strategy_features(utterance)

        if markers:
            utterance.meta[self.marker_attribute_name] = marks
        
        return utterance
    

    def _get_scores(self, corpus: Corpus, selector: Optional[Callable[[Utterance], bool]] = lambda utt: True):
        """
        Calculates average occurrence per utterance. Used in summarize()

        :param corpus: the target Corpus
        :param selector: (lambda) function specifying whether the utterance should be included
        """

        utts = list(corpus.iter_utterances(selector))
        if self.marker_attribute_name not in utts[0].meta:
            print("Could not find politeness markers metadata. Running transform() on corpus first...", end="")
            self.transform(corpus, markers=True)
            print("Done.")

        counts = {k[21:len(k)-2]: 0 for k in utts[0].meta[self.marker_attribute_name].keys()}

        for utt in utts:
            for k, v in utt.meta[self.marker_attribute_name].items():
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
