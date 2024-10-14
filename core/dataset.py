from classifier import Classifier
from csv import DictReader
from os.path import exists
from random import shuffle
from core.data import Data

class Dataset():
    def __init__(self, path: str, classifier: Classifier) -> None:
        self.__path: str = path
        self.__classifier: Classifier = classifier
        self.__data: list[Data] = []
        self.__training_set: list[Data] = []
        self.__test_set: list[Data] = []

        self.__load_data()

    def classify(self) -> None:
        pass

    def get_training_set(self) -> list[Data]:
        return self.__training_set
    
    def get_test_set(self) -> list[Data]:
        return self.__test_set

    def __split_data(self) -> None:
        shuffle(self.__data)

        m: int = len(self.__data) // 3
        self.__training_set = self.__data[m:]
        self.__test_set = self.__data[:m]

    def __load_data(self) -> None:
        if not exists(self.__path):
            raise FileNotFoundError(f"File not found: {self.__path}")
        
        with open(self.__path, 'r') as file:
            reader = DictReader(file)
            data = list(reader)

            self.__data = self.__load(data)
            self.__split_data()

    def __load(self, data: list[Data]):
        if len(data) == 1:
            return [data.clear()]
        
        m: int = len(data) // 2
        left: list[Data] = self.__load(data[:m])
        right: list[Data] = self.__load(data[m:])

        return left + right