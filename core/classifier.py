from abc import ABC, abstractmethod
from data import Data

class Classifier(ABC):
    @abstractmethod
    def classify(self, data: Data, dataset: list[Data] = []) -> int:
        pass

class KNNClassifier(Classifier):
    def __init__(self, k: int = 10) -> None:
        super().__init__()
        self.__k: int = k

    def classify(self, data: Data, dataset: list[Data] = []) -> int:
        distances = [(data.distance(neighbor), neighbor) for neighbor in dataset]

        distances.sort(key=lambda x: x[0])
        k_nearest = distances[:self.__k]

        annotation_counts = {}

        for _, neighbor in k_nearest:
            annotation = neighbor.get_annotation()
            if annotation in annotation_counts:
                annotation_counts[annotation] += 1
            else:
                annotation_counts[annotation] = 1

        return max(annotation_counts, key = annotation_counts.get)