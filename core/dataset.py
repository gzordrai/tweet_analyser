from abc import ABC, abstractmethod
from classifier import Classifier
from csv import reader
from os.path import exists
from random import shuffle
from data import Data

class Dataset(ABC):
    def __init__(self, path: str, classifier: Classifier) -> None:
        self._path: str = path
        self._classifier: Classifier = classifier
        self._data: list[Data] = []

    @abstractmethod
    def classify(self) -> None:
        pass

    def _load_data(self) -> None:
        """
        Load data from a CSV file.
        """

        if not exists(self._path):
            raise FileNotFoundError(f"File not found: {self._path}")
        
        with open(self._path, 'r') as file:
            r = reader(file)
            data = [(line[0], line[-1]) for line in list(r)]

            self._data = self._load(data)

    @abstractmethod
    def _load(self, row: list[dict]) -> list[Data]:
        pass

class AnnotatedDataset(Dataset):
    def __init__(self, path: str, classifier: Classifier) -> None:
        super().__init__(path, classifier)
        self.__training_set: list[Data] = []
        self.__test_set: list[Data] = []
        self._load_data()

    def classify(self) -> int:
        """
        Classify the test set and return the accuracy.
        """

        k: int = 0

        for tweet in self.__test_set:
            print(tweet.get_data())
            annotation: int = self._classifier.classify(tweet, self.__training_set)

            if tweet.get_annotation() == annotation:
                k += 1

        return (k / len(self.__test_set)) * 100

    def _split_data(self) -> None:
        """
        Split the data into a training set and a test set.
        """

        shuffle(self._data)

        m: int = len(self._data) // 3
        self.__training_set = self._data[m:]
        self.__test_set = self._data[:m]

    def _load_data(self):
        """
        Load data from a CSV file.
        """

        super()._load_data()
        self._split_data()

    def _load(self, row) -> list[Data]:
        """
        Load data from a list of rows.
        """

        if len(row) == 1:
            data = Data(*row[0]).clean()

            if data.get_data() == "":
                return []

            return [Data(*row[0]).clean()]
        
        m: int = len(row) // 2
        left: list[dict] = self._load(row[:m])
        right: list[dict] = self._load(row[m:])

        return left + right