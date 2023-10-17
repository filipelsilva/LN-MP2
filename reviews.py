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

def getData():
    with open("./train.txt", "r") as f:
        lines = list(map(splitStringWithClassification, f.readlines()))

    return lines

def main():
    data = getData()

    targets = [d[0] for d in data]
    targets_honesty = [d[0][0].name for d in data]
    targets_polarity = [d[0][1].name for d in data]
    logging.debug(targets)
    logging.debug(targets_honesty)
    logging.debug(targets_polarity)

    count_vect = CountVectorizer()

    # Tokenize the text
    x_train_counts = count_vect.fit_transform([d[1] for d in data])
    logging.debug(x_train_counts.shape)

    # Apply tf-idf
    tfidf_transformer = TfidfTransformer()
    x_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)
    logging.debug(x_train_tfidf.shape)

    # Train a classifier
    for t in [targets_honesty, targets_polarity]:
        clf = MultinomialNB().fit(x_train_tfidf, t)

        docs_new = ['God is love', 'OpenGL on the GPU is fast']
        x_new_counts = count_vect.transform(docs_new)
        x_new_tfidf = tfidf_transformer.transform(x_new_counts)

        predicted = clf.predict(x_new_tfidf)

        for doc, category in zip(docs_new, predicted):
            logging.info('%r => %s' % (doc, category))

if __name__ == "__main__":
    main()
