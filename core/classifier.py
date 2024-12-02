from abc import ABC, abstractmethod
from collections import defaultdict
from numpy import array, fromiter, log, ndarray

from data import Data

class Classifier(ABC):
    @abstractmethod
    def classify(self, data: Data, dataset: ndarray[Data] = array([])) -> int:
        pass

class KNNClassifier(Classifier):
    def __init__(self, k: int = 10) -> None:
        super().__init__()
        self.__k: int = k

    def set_k(self, k: int):
        self.__k = k

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
    def __init__(self) -> None:
        self.class_probabilities = defaultdict(float)
        self.feature_probabilities = defaultdict(lambda: defaultdict(float))
        self.classes = set()
        self.vocab = set()

    def _generate_ngrams(self, words, n=1):
        """Generate n-grams from a list of words."""
        if not isinstance(words, list):
            raise TypeError(f"Expected list of words, got {type(words)}")
        if n < 1:
            raise ValueError("ngram_size must be 1 or greater")

        if n == 1:
            return words
        return [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]

    def train(self, dataset, ngram_size=1) -> None:
        class_counts = defaultdict(int)
        feature_counts = defaultdict(lambda: defaultdict(int))
        total_count = len(dataset)

        for data in dataset:
            annotation = data.get_annotation()
            self.classes.add(annotation)
            class_counts[annotation] += 1

            words = [word for word in data.get_data().split() if len(word) > 3]
            ngrams = self._generate_ngrams(words, ngram_size)

            for ngram in ngrams:
                self.vocab.add(ngram)
                feature_counts[annotation][ngram] += 1

        for annotation in self.classes:
            self.class_probabilities[annotation] = class_counts[annotation] / total_count
            total_ngrams = sum(feature_counts[annotation].values())

            for ngram in feature_counts[annotation]:
                self.feature_probabilities[annotation][ngram] = (
                    feature_counts[annotation][ngram] + 1
                ) / (total_ngrams + len(self.vocab))

    def classify(self, data: Data, ngram_size=1) -> str:
        words = [word for word in data.get_data().split() if len(word) > 3]
        ngrams = self._generate_ngrams(words, ngram_size)

        class_scores = defaultdict(float)

        for annotation in self.classes:
            class_scores[annotation] = log(self.class_probabilities[annotation])

            for ngram in ngrams:
                if ngram in self.feature_probabilities[annotation]:
                    class_scores[annotation] += log(self.feature_probabilities[annotation][ngram])
                else:
                    # Apply Laplace smoothing for unseen n-grams
                    class_scores[annotation] += log(
                        1 / (sum(self.feature_probabilities[annotation].values()) + len(self.vocab))
                    )

        return max(class_scores, key=class_scores.get)