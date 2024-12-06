from abc import ABC, abstractmethod
from os.path import exists
from numpy import array, fromiter, loadtxt, zeros, ndarray
from random import shuffle
from time import time
from tqdm import tqdm

import pandas as pd
from pandas import DataFrame

from .classifier import Classifier
from .data import Data

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
        print(f"Loading data from {self._path}...")
        start = time()

        if not exists(self._path):
            raise FileNotFoundError(f"File not found: {self._path}")
        
        data = loadtxt(
            self._path,
            delimiter = ',',
            dtype = str,
            usecols = (0, -1),
            skiprows = 1
            )
        self._data = fromiter(
            (Data(*row).clean() for row in data),
            dtype = Data,
            count = len(data)
            )

        print(f"Data loaded in {time() - start:.2f} seconds.")

        self._split_data()
    
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
        self._test_set: ndarray[Data] = self._data
        self._training_set: ndarray[Data] = dataset.get_data()
        self._annotated_df = pd.DataFrame(columns=[0, 1])

    def _split_data(self) -> None:
        self._test_set = self._data

    def get_df(self, annotation, tweet) -> DataFrame :
        self._annotated_df = pd.concat([self._annotated_df, pd.DataFrame([[annotation, tweet]], columns=[0, 1])], ignore_index=True)

    def classify(self) -> int:
        """
        Classify the test set and return the accuracy.
        """
        k: int = 0

        for i in tqdm(range(len(self._test_set))):
            tweet: Data = self._test_set[i]
            annotation: int = self._classifier.classify(tweet, self._training_set)

            if annotation == tweet.get_annotation():
                k += 1

            tweet.set_annotation(annotation)
            clean_annotation = str(annotation).strip('"')
            self.get_df(int(clean_annotation), tweet.get_data())

        return ((k / len(self._test_set)) * 100),self._annotated_df
