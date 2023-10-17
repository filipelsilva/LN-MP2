import logging
import sys
from enum import Enum

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(levelname)s[%(lineno)d]: %(message)s')

class Honesty(Enum):
    TRUTHFUL = 1
    DECEPTIVE = 2

    @staticmethod
    def get(string):
        if Honesty.TRUTHFUL.name in string:
            return Honesty.TRUTHFUL
        return Honesty.DECEPTIVE

class Polarity(Enum):
    POSITIVE = 1
    NEGATIVE = 2

    @staticmethod
    def get(string):
        if Polarity.POSITIVE.name in string:
            return Polarity.POSITIVE
        return Polarity.NEGATIVE

def splitStringWithClassification(string):
    split = string.strip().split("\t")
    return [[Honesty.get(split[0]), Polarity.get(split[0])], split[1]]

def getTrainFileLines():
    with open("./train.txt", "r") as f:
        lines = list(map(splitStringWithClassification, f.readlines()))

    return lines

def getTestFileLines():
    with open("./test_just_reviews.txt", "r") as f:
        lines = list(map(lambda s: s.strip(), f.readlines()))

    return lines

def main():
    train_lines = getTrainFileLines()
    test_lines = getTestFileLines()

    data = [l[1] for l in train_lines]
    targets = [l[0] for l in train_lines]
    targets_honesty = [l[0][0].name for l in train_lines]
    targets_polarity = [l[0][1].name for l in train_lines]
    logging.debug(targets)
    logging.debug(targets_honesty)
    logging.debug(targets_polarity)

    text_clf = Pipeline([
        ('vect-tfidf', TfidfVectorizer()),
        ('clf', MultinomialNB()),
    ])

    for t in [targets_honesty, targets_polarity]:
        clf = text_clf.fit(data, t)

        predicted = clf.predict(test_lines)

        for doc, category in zip(test_lines, predicted):
            logging.info('%r => %s' % (doc, category))

if __name__ == "__main__":
    main()
