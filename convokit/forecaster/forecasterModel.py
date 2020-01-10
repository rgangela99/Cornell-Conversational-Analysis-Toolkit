from abc import ABC, abstractmethod


class ForecasterModel(ABC):

    def __init__(self, forecast_feat_name: str = "prediction", forecast_prob_feat_name: str = "score"):
        """

        :param forecast_feat_name: name for DataFrame column containing predictions, default: "prediction"
        :param forecast_prob_feat_name: name for column containing prediction scores, default: "score"
        """
        self.forecast_feat_name = forecast_feat_name
        self.forecast_prob_feat_name = forecast_prob_feat_name

    @abstractmethod
    def train(self, id_to_context_reply_label):
        """
        Train the Forecaster Model with the context-reply-label tuples
        """
        pass

    @abstractmethod
    def forecast(self, id_to_context_reply_label):
        """
        Use the Forecaster Model to compute forecasts and scores
        for given context-reply pairs and return a results dataframe
        """
        pass


