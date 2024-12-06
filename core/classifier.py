from abc import ABC, abstractmethod
from collections import defaultdict
from numpy import array, fromiter, log, ndarray

from .data import Data

class Classifier(ABC):
    @abstractmethod
    def classify(self, data: Data, dataset: ndarray[Data] = array([])) -> int:
        pass

class KNNClassifier(Classifier):
    def __init__(self, k: int = 10) -> None:
        super().__init__()
        self.__k: int = k

    def classify(self, data: Data, dataset: ndarray[Data] = array([])) -> int:
        distances = fromiter(
            ((data.distance(neighbor), neighbor) for neighbor in dataset),
            dtype=object,
            count=len(dataset),
        )

        k_nearest = sorted(distances, key=lambda x: x[0])[: self.__k]
        annotation_counts = defaultdict(int)

        for _, neighbor in k_nearest:
            annotation = neighbor.get_annotation()
            if annotation in annotation_counts:
                annotation_counts[annotation] += 1
            else:
                annotation_counts[annotation] = 1

        return max(annotation_counts, key=annotation_counts.get)

class NaiveBayesClassifier(Classifier):
    def __init__(self, ngram_size: int = 1) -> None:
        self.__class_probabilities = defaultdict(float)
        self.__feature_probabilities = defaultdict(lambda: defaultdict(float))
        self.__classes = set()
        self.__vocab = set()
        self.__ngram_size: int = ngram_size
        
    def _generate_ngrams(self, words):
        """
        Generate n-grams from a list of words.
        """
        print(words)

        if self.__ngram_size == 1:
            return words

        return [" ".join(words[i:i + self.__ngram_size]) for i in range(len(words) - self.__ngram_size + 1)]

    def set_ngram_size(self, size: int) -> None:
        self.__ngram_size = size

    def train(self, dataset) -> None:
        class_counts = defaultdict(int)
        feature_counts = defaultdict(lambda: defaultdict(int))
        total_count = len(dataset)

        for data in dataset:
            annotation = data.get_annotation()
            self.__classes.add(annotation)
            class_counts[annotation] += 1

            words = [word for word in data.get_data().split() if len(word) > 3]
            ngrams = self._generate_ngrams(words)

            for ngram in ngrams:
                self.__vocab.add(ngram)
                feature_counts[annotation][ngram] += 1

        for annotation in self.__classes:
            self.__class_probabilities[annotation] = class_counts[annotation] / total_count
            total_ngrams = sum(feature_counts[annotation].values())

            for ngram in feature_counts[annotation]:
                self.__feature_probabilities[annotation][ngram] = (
                    feature_counts[annotation][ngram] + 1
                ) / (total_ngrams + len(self.__vocab))

    def classify(self, data: Data, dataset = array([])) -> str:
        if not self.__classes:
            raise ValueError("Classifier has not been trained yet.")

        words = [word for word in data.get_data().split() if len(word) > 3]
        ngrams = self._generate_ngrams(words)

        class_scores = defaultdict(float)

        for annotation in self.__classes:
            class_scores[annotation] = log(self.__class_probabilities[annotation])

            for ngram in ngrams:
                if ngram in self.__feature_probabilities[annotation]:
                    class_scores[annotation] += log(self.__feature_probabilities[annotation][ngram])
                else:
                    class_scores[annotation] += log(
                        1 / (sum(self.__feature_probabilities[annotation].values()) + len(self.__vocab))
                    )

        return max(class_scores, key=class_scores.get)