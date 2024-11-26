from numpy import array
from re import sub
from data_distance import DistanceStrategy, WordOverlapDistance

class Data():
    def __init__(self, annotation: str, data: str, distance: DistanceStrategy = WordOverlapDistance()) -> None:
        self.__annotation = annotation
        self.__data: str = data
        self.__distance: DistanceStrategy = distance
        self.__words = array(data.split(' '))
        self.__length = len(self.__words)
        self.__set_words = set(self.__words)

    def clean(self):
        patterns = [
            (r"@[a-zA-Z0-9]+", ""),                                                                         # Remove mentions
            (r"#[a-zA-Z0-9]+", ""),                                                                         # Remove hashtags
            (r"RT", ""),                                                                                    # Remove retweets
            (r".+ - http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", ""), # Remove attached links
            (r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", ""),      # Remove links
            (r"\[.*?\]", ""),                                                                               # Remove square brackets
            (r".*[:;]\[\)][^\n]*[:;][\(].*|.*[:;][\(][^\n]*[:;][\)].*", ""),                                 # Remove happy and sad emoticons in the same tweet
            (r"(?<=[a-zA-Z])[!\?\"\.;,]", r" \g<0>"),                                                       # Add space before punctuation only if there's a letter before
            (r"[.,!?]", ""),                                                                                # Remove periods and commas
            (r"[!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~]", ""),                                                  # Remove punctuation
            (r"([a-zA-Z])\1{2,}", r"\1\1"),                                                                 # Limit letter repetitions
            (r"\s+", " ")                                                                                   # Replace multiple spaces with a single space
        ]

        for pattern in patterns:
            if self.__data == "":
                break

            self.__data = sub(pattern[0], pattern[1], self.__data)

        return Data(self.__annotation, self.__data.lower().strip())

    def get_words(self) -> array:
        return self.__words
    
    def get_set_words(self) -> set:
        return self.__set_words

    def __len__(self) -> int:
        return self.__length

    def distance(self, d) -> int:
        return self.__distance.distance(self, d)
    
    def set_annotation(self, annotation: str) -> None:
        self.__annotation = annotation

    def get_annotation(self) -> str:
        return self.__annotation

    def get_data(self) -> str:
        return self.__data
