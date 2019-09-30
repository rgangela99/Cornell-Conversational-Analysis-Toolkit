from abc import ABC, abstractmethod

class Framework(ABC):

    @abstractmethod
    def fit(self, *args):
        pass

    @abstractmethod
    def predict(self, *args):
        pass

    def fit_predict(self, *args):
        pass