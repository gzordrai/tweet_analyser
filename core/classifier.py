from abc import ABC, abstractmethod
from data import Data
from data_distance import DistanceStrategy, WordOverlapDistance

class Classifier(ABC):
    def __init__(self, strategy: DistanceStrategy = WordOverlapDistance) -> None:
        self.__distance_strategy: DistanceStrategy = strategy

    def set_distance_strategy(self, strategy: DistanceStrategy) -> None:
        self.__distance_strategy = strategy
    
    @abstractmethod
    def classify(self, data: Data, dataset: list[Data] = []) -> int:
        pass

class KNNClassifier(Classifier):
    def __init__(self, k = 10) -> None:
        super().__init__()
        self.__k = k

    def classify(self, data: Data, dataset: list[Data] = []) -> int:
        neighbors: list[Data] = dataset[:self.__k]
        distances: list[int] = [data.distance(neighbor) for neighbor in neighbors]

        for d in dataset[self.__k:]:
            distance = data.distance(d)

            if distance < max(distances):
                neighbors[distances.index(max(distances))] = d
                distances[distances.index(max(distances))] = distance

        return max(set(neighbors), key = neighbors.count).get_annotation()