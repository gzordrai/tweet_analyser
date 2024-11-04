from abc import ABC, abstractmethod

class DistanceStrategy(ABC):
    @abstractmethod
    def distance(self, source, target) -> int:
        pass

class WordOverlapDistance(DistanceStrategy):
    def distance(self, source, target) -> int:
        s: str = source.get_data().split(' ')
        t: str = target.get_data().split(' ')
        total_words: int = len(s) + len(t)
        common_words: int  = 0

        for word in source.get_data().split(' '):
            if word in target.get_data():
                common_words += 1

        for word in target.get_data().split(' '):
            if word in source.get_data():
                common_words += 1

        return (total_words - common_words) / total_words