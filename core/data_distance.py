from abc import ABC, abstractmethod
from numpy import array, intersect1d

class DistanceStrategy(ABC):
    @abstractmethod
    def distance(self, source, target) -> int:
        pass

class WordOverlapDistance(DistanceStrategy):
    def distance(self, source, target) -> int:
        total_words = len(source) + len(target)
        common_words = len(source.get_set_words().intersection(target.get_set_words()))
        result = (total_words - 2 * common_words) / total_words

        return result