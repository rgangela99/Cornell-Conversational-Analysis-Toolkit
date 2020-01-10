from convokit.model import Corpus, Conversation, Utterance
from typing import Callable, Optional
from convokit import Transformer
from .cumulativeBoW import CumulativeBoW
from .forecasterModel import ForecasterModel
import pandas as pd

class Forecaster(Transformer):
    """
    Implements basic Forecaster behavior
    """
    def __init__(self, forecaster_model: ForecasterModel = None,
                 convo_structure: str = "branched",
                 text_func=lambda utt: utt.text,
                 convo_selector_func: Callable[[Conversation], bool] = lambda convo: True,
                 utt_selector_func: Callable[[Utterance], bool] = lambda utt: True,
                 label_feat: Optional[str] = None,
                 forecast_feat_name: str = "forecast", forecast_prob_feat_name: str = "forecast_prob",
                 skip_broken_convos: bool = False):
        """

        :param forecaster_model: ForecasterModel to use, e.g. cumulativeBoW or CRAFT
        :param convo_structure: conversations in expected corpus are 'branched' or 'linear', default: "branched"
        :param text_func: optional function for extracting the text of the utterance, default: use utterance text
        :param convo_selector_func: function for selecting for the conversations to analyze, default: use all conversations
        :param utt_selector_func: function for selecting for the utterances in selected conversations to analyze, default: use all utterances
        :param label_feat: optional name of utterance metadata feature containing the utterance's forecast label; only used in training
        :param forecast_feat_name: metadata feature name to use in annotation for forecast result, default: "forecast"
        :param forecast_prob_feat_name: metadata feature name to use in annotation for forecast result probability, default: "forecast_prob"
        :param skip_broken_convos: if True, exclude all conversations that have broken reply-to structures
        """

        assert convo_structure in ["branched", "linear"]
        self.convo_structure = convo_structure

        if forecaster_model is None:
            print("No model passed to Forecaster. Initializing default forecaster model: Cumulative Bag-of-words...")
            self.forecaster_model = CumulativeBoW(forecast_feat_name=forecast_feat_name,
                                                  forecast_prob_feat_name=forecast_prob_feat_name)
        else:
            self.forecaster_model = forecaster_model

        self.label_feat = label_feat
        self.text_func = text_func
        self.utt_selector_func = utt_selector_func
        self.convo_selector_func = convo_selector_func
        self.forecast_feat_name = forecast_feat_name
        self.forecast_prob_feat_name = forecast_prob_feat_name
        self.skip_broken_convos = skip_broken_convos

    def _get_context_reply_label_dict(self, corpus: Corpus, include_label=True):
        dialogs = []
        if self.convo_structure == "branched":
            for convo in corpus.iter_conversations(self.convo_selector_func):
                try:
                    for path in convo.get_root_to_leaf_paths():
                        path = [utt for utt in path if self.utt_selector_func(utt)]
                        if len(path) == 1: continue
                        dialogs.append(path)
                except ValueError as e:
                    if not self.skip_broken_convos:
                        raise e

        elif self.convo_structure == "linear":
            for convo in corpus.iter_conversations(self.convo_selector_func):
                utts = convo.get_chronological_utterance_list(selector=self.utt_selector_func)
                if len(utts) == 1: continue
                dialogs.append(utts)

        id_to_context_reply_label = dict()

        for dialog in dialogs:
            for idx in range(1, len(dialog)):
                reply = self.text_func(dialog[idx])
                label = dialog[idx].meta[self.label_feat] if include_label else None
                reply_id = dialog[idx].id
                context = [self.text_func(utt) for utt in dialog[:idx]]
                id_to_context_reply_label[reply_id] = (context, reply, label) if include_label else (context, reply, None)

        return id_to_context_reply_label

    def fit(self, corpus: Corpus, y=None):
        """
        Train the ForecasterModel on the given corpus
        """
        id_to_context_reply_label = self._get_context_reply_label_dict(corpus, include_label=True)
        self.forecaster_model.train(id_to_context_reply_label)

    def transform(self, corpus: Corpus) -> Corpus:
        """
        Annotate the corpus utterances with forecast and forecast probability information
        :param corpus: target Corpus
        :return: annotated Corpus
        """
        id_to_context_reply_label = self._get_context_reply_label_dict(corpus, include_label=False)
        forecast_df = self.forecaster_model.forecast(id_to_context_reply_label)

        for utt in corpus.iter_utterances():
            if utt.id in forecast_df.index:
                utt.add_meta(self.forecast_feat_name, forecast_df.loc[utt.id][self.forecast_feat_name])
                utt.add_meta(self.forecast_prob_feat_name, forecast_df.loc[utt.id][self.forecast_prob_feat_name])
            else:
                utt.add_meta(self.forecast_feat_name, None)
                utt.add_meta(self.forecast_prob_feat_name, None)

        return corpus

    def summarize(self, corpus: Corpus, use_selector=True, exclude_na=True):
        """
        Returns a DataFrame of utterances and their forecasts (and forecast probabilities)
        :param corpus: target Corpus
        :param use_selector: whether to use Forecaster's convo and utterance selector functions
        :param exclude_na: whether to drop NaN results
        :return: a pandas DataFrame
        """
        utt_forecast_prob = []
        if use_selector:
            for convo in corpus.iter_conversations(self.convo_selector_func):
                for utt in convo.iter_utterances(self.utt_selector_func):
                    utt_forecast_prob.append((utt.id, utt.meta[self.forecast_feat_name], utt.meta[self.forecast_prob_feat_name]))
        else:
            for utt in corpus.iter_utterances():
                utt_forecast_prob.append((utt.id, utt.meta[self.forecast_feat_name], utt.meta[self.forecast_prob_feat_name]))
        forecast_df = pd.DataFrame(utt_forecast_prob, columns=["utt_id", self.forecast_feat_name, self.forecast_prob_feat_name]) \
            .set_index('utt_id').sort_values(self.forecast_prob_feat_name, ascending=False)
        if exclude_na:
            forecast_df = forecast_df.dropna()
        return forecast_df

    def get_model(self):
        """
        Get the forecaster model object
        """
        return self.forecaster_model

    def set_model(self, forecaster_model):
        """
        Set the forecaster model
        :return:
        """
        self.forecaster_model = forecaster_model
