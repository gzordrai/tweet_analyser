from dataset import AnnotatedDataset
from classifier import Classifier, KNNClassifier
from os import getcwd
from os.path import join

if __name__ == "__main__":
    path: str = join(getcwd(), "datasets/inputs/testdata.manual.2009.06.14.csv")
    classifier: Classifier = KNNClassifier(10)
    accuracy: float = 0.0

    dataset: AnnotatedDataset = AnnotatedDataset(path, classifier)
    print(f"Accurency {dataset.classify()}%")