from abc import ABC, abstractmethod
from classifier import Classifier
from csv import reader
from os.path import exists
from numpy import array, ndarray
from random import shuffle
from tqdm import tqdm
from data import Data

class Dataset(ABC):
    def __init__(self, path: str, classifier: Classifier) -> None:
        self._path: str = path
        self._classifier: Classifier = classifier
        self._data: ndarray = array([])
        self._training_set: ndarray = array([])
        self._test_set: ndarray = array([])

        self._load_data()

    @abstractmethod
    def classify(self) -> int:
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

            self._data = array(self._load(data))
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
        left: list[Data] = self._load(row[:m])
        right: list[Data] = self._load(row[m:])

        return left + right
    
    @abstractmethod
    def _split_data(self) -> None:
        pass
    
    def get_data(self) -> ndarray:
        return self._data
    
    def save(self) -> None:
        annotated_file_path: str = "annotated.csv"

        with open(annotated_file_path, 'w') as file:
            for tweet in self._data:
                file.write(f"{tweet.get_annotation()},{tweet.get_data()}\n")

class AnnotatedDataset(Dataset):
    def __init__(self, path: str, classifier: Classifier) -> None:
        super().__init__(path, classifier)

    def _split_data(self) -> None:
        """
        Split the data into a training set and a test set.
        """

        shuffle(self._data)

        m: int = len(self._data) // 3
        self._training_set = self._data[m:]
        self._test_set = self._data[:m]
    
    def classify(self) -> int:
        """
        Classify the test set and return the accuracy.
        """
        k: int = 0

        for i in tqdm(range(len(self._test_set))):
            tweet = self._test_set[i]
            annotation: int = self._classifier.classify(tweet, self._training_set)

            if tweet.get_annotation() == annotation:
                k += 1

        return (k / len(self._test_set)) * 100

class UnannotateDataset(Dataset):
    def __init__(self, path: str, classifier: Classifier, dataset: AnnotatedDataset) -> None:
        super().__init__(path, classifier)
        self._test_set: list[Data] = self._data
        self._training_set: list[Data] = dataset.get_data()

    def _split_data(self) -> None:
        self._test_set = self._data

    def classify(self) -> int:
        """
        Classify the test set and return the accuracy.
        """

        print("Classifying the test set...")

        for i in tqdm(range(len(self._test_set))):
            tweet: Data = self._test_set[i]
            annotation: int = self._classifier.classify(tweet, self._training_set)

            tweet.set_annotation(annotation)
