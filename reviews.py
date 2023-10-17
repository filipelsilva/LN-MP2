import logging
import sys

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(levelname)s[%(lineno)d]: %(message)s'
)

HONESTY_CLASSIFICATION = ["TRUTHFUL", "DECEPTIVE"]
POLARITY_CLASSIFICATION = ["POSITIVE", "NEGATIVE"]


def getHonesty(string):
    for h in HONESTY_CLASSIFICATION:
        if h in string:
            return h


def getPolarity(string):
    for p in POLARITY_CLASSIFICATION:
        if p in string:
            return p


def splitStringWithClassification(string):
    split = string.strip().split("\t")
    return [[getHonesty(split[0]), getPolarity(split[0])], split[1]]


def getTrainFileLines():
    with open("./train.txt", "r") as f:
        lines = list(map(splitStringWithClassification, f.readlines()))

    return lines


def getTestFileLines():
    with open("./test_just_reviews.txt", "r") as f:
        lines = list(map(lambda s: s.strip(), f.readlines()))

    return lines


def writeResults(results):
    logging.debug("WRITING RESULTS: " + results)
    with open("./results.txt", "w") as f:
        for r in results:
            f.write(r + "\n")


def optimizeParameters(model, data, targets):
    parameters = {
        # 'vect-tfidf__ngram_range': [(1, 1), (1, 2), (1, 3), (1, 4)],
        # 'clf__penalty': ('l1', 'l2', 'elasticnet', None),
        # 'clf__alpha': (1e-2, 1e-3, 1e-4, 1e-5, 1e-6),
        'clf__tol': (None, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6),
        'clf__max_iter': (1000, 100000, 1000000)
    }

    gs_clf = GridSearchCV(model, parameters, cv=5, n_jobs=-1)
    gs_clf = gs_clf.fit(data, targets)
    for param_name in sorted(parameters.keys()):
        logging.info("%s: %r" % (param_name, gs_clf.best_params_[param_name]))

    exit(0)


def main():
    train_lines = getTrainFileLines()
    test_lines = getTestFileLines()

    data = [l[1] for l in train_lines]
    targets = [l[0][0] + l[0][1] for l in train_lines]
    logging.debug(targets)

    text_clf = Pipeline([
        ('vect-tfidf', TfidfVectorizer(
            use_idf=True,
            ngram_range=(1, 2),
        )),
        ('clf', SGDClassifier(
            loss='hinge',
            penalty='elasticnet',
            alpha=0.0001,
            tol=None,
            max_iter=1000,
            random_state=42, # For reproducibility
        ))
    ])

    if sys.argv[1] == "train":
        optimizeParameters(text_clf, data, targets)

    data_train, data_test, targets_train, targets_test = train_test_split(
        data,
        targets,
        test_size=0.25,
        random_state=42 # For reproducibility TODO remove
    )

    clf = text_clf.fit(data_train, targets_train)
    predicted = clf.predict(data_test)

    # Print the classification report
    logging.info(metrics.classification_report(targets_test, predicted))

    # Print and plot the confusion matrix
    cm = metrics.confusion_matrix(targets_test, predicted)
    print(cm)

    for doc, category in zip(test_lines, predicted):
        logging.debug('%r => %s' % (doc, category))


if __name__ == "__main__":
    main()
