from abc import ABC, abstractmethod
from convokit.model import Corpus

class Framework(ABC):

    def fit(self, *args) -> None:
        pass

    @abstractmethod
    def predict(self, *args) -> None:

        pass

    def fit_predict(self, *args) -> None:
        pass