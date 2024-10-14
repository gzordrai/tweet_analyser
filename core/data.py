from re import sub
from core.data_distance import DistanceStrategy, WordOverlapDistance

class Data():
    def __init__(self, data: str, distance: DistanceStrategy = WordOverlapDistance) -> None:
        self.__data: str = data
        self.__distance: DistanceStrategy = distance(data)

    def clean(self):
        patterns = [
            (r"@[a-zA-Z0-9]+", ""),                                                                         # Remove mentions
            (r"#[a-zA-Z0-9]+", ""),                                                                         # Remove hashtags
            (r"RT", ""),                                                                                    # Remove retweets
            (r".+ - http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", ""), # Remove attached links
            (r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", ""),      # Remove links
            (r".*[:;][\)][^\n]*[:;][\(].*|.*[:;][\(][^\n]*[:;][\)].*", ""),                                 # Remove happy and sad emoticons in the same tweet
            (r"(?<=[a-zA-Z])[!\?\"\.;,]", r" \g<0>"),                                                       # Add space before punctuation only if there's a letter before
            (r"[!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]", ""),                                                 # Remove punctuation
            (r"[.,]", ""),                                                                                  # Remove periods and commas
            (r"\s+", " "),                                                                                  # Replace multiple spaces with a single space
        ]

        for pattern in patterns:
            if self.__data == "":
                break

            self.__data = sub(pattern[0], pattern[1], self.__data)

        return Data(self.__data.lower())

    def get_data(self) -> str:
        return self.__data

    def distance(self, d) -> int:
        return self.__distance.distance(self, d)
