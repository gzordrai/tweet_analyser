from os import getcwd
from os.path import join

from dataset import AnnotatedDataset, UnannotateDataset
from classifier import Classifier, KNNClassifier, NaiveBayesClassifier

if __name__ == "__main__":
    annotated_dataset_path: str = join(getcwd(), "datasets/inputs/testdata.manual.2009.06.14.csv")
    unannotated_dataset_path: str = join(getcwd(), "datasets/inputs/training.1600000.processed.noemoticon.csv")
    knn_classifier: Classifier = KNNClassifier(10)
    bayes_classifier: Classifier = NaiveBayesClassifier(3)
    annotated_dataset: AnnotatedDataset = AnnotatedDataset(annotated_dataset_path, bayes_classifier)
    unannotated_dataset: UnannotateDataset = UnannotateDataset(unannotated_dataset_path, bayes_classifier, annotated_dataset)

    bayes_classifier.train(unannotated_dataset.get_data())
    print(unannotated_dataset.classify(), "%")