from abc import ABC, abstractmethod
from core.data_distance import DistanceStrategy, WordOverlapDistance

class Classifier(ABC):
    def __init__(self, strategy: DistanceStrategy = WordOverlapDistance) -> None:
        self.__distance_strategy: DistanceStrategy = strategy

    def set_distance_strategy(self, strategy: DistanceStrategy) -> None:
        self.__distance_strategy = strategy
    
    @abstractmethod
    def classify(self, data) -> str:
        pass

class KNNClassifier(Classifier):
    def __init__(self, k = 3):
        self.__k = k

    def classify(self, data) -> str:
        pass

    