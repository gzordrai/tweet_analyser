from abc import ABC, abstractmethod
from collections import defaultdict
from numpy import log
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

class NaiveBayesClassifier(Classifier):
    def __init__(self) -> None:
        super().__init__()
        self.class_probabilities = defaultdict(float)
        self.feature_probabilities = defaultdict(lambda: defaultdict(float))
        self.classes = set()

    def train(self, dataset: list[Data]) -> None:
        class_counts = defaultdict(int)
        feature_counts = defaultdict(lambda: defaultdict(int))
        total_count = len(dataset)

        for data in dataset:
            annotation = data.get_annotation()
            self.classes.add(annotation)
            class_counts[annotation] += 1

            words = data.get_data().split()

            for word in words:
                feature_counts[annotation][word] += 1

        for annotation in self.classes:
            self.class_probabilities[annotation] = class_counts[annotation] / total_count
            total_words = sum(feature_counts[annotation].values())

            for word in feature_counts[annotation]:
                self.feature_probabilities[annotation][word] = (feature_counts[annotation][word] + 1) / (total_words + len(feature_counts[annotation]))

    def classify(self, data: Data, dataset: list[Data] = []) -> int:
        words = data.get_data().split()
        class_scores = defaultdict(float)

        for annotation in self.classes:
            class_scores[annotation] = log(self.class_probabilities[annotation])

            for word in words:
                if word in self.feature_probabilities[annotation]:
                    class_scores[annotation] += log(self.feature_probabilities[annotation][word])
                else:
                    class_scores[annotation] += log(1 / (sum(self.feature_probabilities[annotation].values()) + len(self.feature_probabilities[annotation])))

        return max(class_scores, key=class_scores.get)