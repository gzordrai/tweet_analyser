from abc import ABC, abstractmethod

class DistanceStrategy(ABC):
    def __init__(self, data: str) -> None:
        pass

    @abstractmethod
    def distance(self, source, target) -> int:
        pass

class WordOverlapDistance(DistanceStrategy):
    def distance(self, source, target) -> int:
        data: str = target.get_data()
        i, j = len(data.split(' ')), 0

        for word in source.get_data().split(' '):
            i += 1

            if word in data:
                j += 1

        return (i - j) / i
