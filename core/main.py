from dataset import Dataset
from classifier import Classifier, KNNClassifier
from os import getcwd
from os.path import join

if __name__ == "__main__":
    path: str = join(getcwd(), "datasets/inputs/training.1600000.processed.noemoticon.csv")
    classifier: Classifier = KNNClassifier(10)
    dataset: Dataset = Dataset(path, classifier)