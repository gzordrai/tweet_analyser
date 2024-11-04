from dataset import AnnotatedDataset, UnannotateDataset
from classifier import Classifier, KNNClassifier
from os import getcwd
from os.path import join
from tqdm import tqdm

if __name__ == "__main__":
    # annotated_dataset_path: str = join(getcwd(), "datasets/inputs/testdata.manual.2009.06.14.csv")
    annotated_dataset_path: str = join(getcwd(), "annotated.csv")
    unannotated_dataset_path: str = join(getcwd(), "datasets/inputs/training.1600000.processed.noemoticon.csv")
    classifier: Classifier = KNNClassifier(10)
    accuracy: float = 0.0

    annotated_dataset: AnnotatedDataset = AnnotatedDataset(annotated_dataset_path, classifier)

    print(f"Accurency {annotated_dataset.classify()}%")

    #unannotated_dataset: UnannotateDataset = UnannotateDataset(unannotated_dataset_path, classifier, annotated_dataset)

    # unannotated_dataset.classify()
    # unannotated_dataset.save()

    # k: int = 100

    # for _ in tqdm(range(k)):
    #     dataset = AnnotatedDataset(annotated_dataset_path, classifier)
    #     accuracy += dataset.classify()

    # print(f"Accurency {accuracy / k}%")