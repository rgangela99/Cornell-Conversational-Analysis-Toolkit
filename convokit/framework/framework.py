from abc import ABC, abstractmethod
from convokit.model import Corpus

class Framework(ABC):

    def fit(self, *args):
        pass

    @abstractmethod
    def predict(self, *args):

        pass

    def fit_predict(self, *args):
        pass