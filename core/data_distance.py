from abc import ABC, abstractmethod
from numpy import array, intersect1d

class DistanceStrategy(ABC):
    @abstractmethod
    def distance(self, source, target) -> int:
        pass

class WordOverlapDistance(DistanceStrategy):
    def distance(self, source, target) -> int:
        s = array(source.get_data().split(' '))
        t = array(target.get_data().split(' '))
        total_words = len(s) + len(t)

        common_words = len(intersect1d(s, t))

        return (total_words - 2 * common_words) / total_words