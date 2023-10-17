import logging
import sys
from enum import Enum

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
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

def getLines():
    with open("./train.txt", "r") as f:
        lines = list(map(splitStringWithClassification, f.readlines()))

    return lines

def main():
    lines = getLines()

    data = [l[1] for l in lines]
    targets = [l[0] for l in lines]
    targets_honesty = [l[0][0].name for l in lines]
    targets_polarity = [l[0][1].name for l in lines]
    logging.debug(targets)
    logging.debug(targets_honesty)
    logging.debug(targets_polarity)

    text_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB()),
    ])

    for t in [targets_honesty, targets_polarity]:
        clf = text_clf.fit(data, t)

        docs_new = ['God is love', 'OpenGL on the GPU is fast']

        predicted = clf.predict(docs_new)

        for doc, category in zip(docs_new, predicted):
            logging.info('%r => %s' % (doc, category))

if __name__ == "__main__":
    main()
