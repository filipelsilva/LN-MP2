import logging
import sys

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn import metrics

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='===== %(levelname)s[%(lineno)d] =====\n%(message)s\n'
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
    logging.debug("WRITING RESULTS")
    logging.debug(results)
    with open("./results.txt", "w") as f:
        for r in results:
            f.write(r + "\n")


def optimizeParameters(model, data, targets):
    parameters = {
        # 'vect-tfidf__use_idf': (True, False),
        # 'vect-tfidf__ngram_range': [(1, 1), (1, 2), (1, 3), (1, 4)],
        'clf__estimator__loss': ('hinge', 'log_loss', 'modified_huber', 'squared_hinge', 'perceptron', 'squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'),
        'clf__estimator__penalty': ('l1', 'l2', 'elasticnet', None),
        'clf__estimator__alpha': (1e-2, 1e-3, 1e-4, 1e-5, 1e-6),
        'clf__estimator__tol': (None, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6),
    }

    gs_clf = GridSearchCV(model, parameters, cv=5, n_jobs=-4)
    gs_clf = gs_clf.fit(data, targets)
    for param_name in sorted(parameters.keys()):
        logging.info("%s: %r" % (param_name, gs_clf.best_params_[param_name]))


def getResults(predicted, targets):
    # Print the classification report
    logging.info(metrics.classification_report(targets, predicted))
    # logging.info(metrics.classification_report(targets, predicted, output_dict=True)['accuracy'])

    # Print and plot the confusion matrix
    cm = metrics.confusion_matrix(targets, predicted)
    logging.info(cm)


def testOnTrainSet(model, data, targets):
    data_train, data_test, targets_train, targets_test = train_test_split(
        data,
        targets,
        test_size=0.20,
    )

    clf = model.fit(data_train, targets_train)
    predicted = clf.predict(data_test)

    # As multioutput classifier is still not supported by metrics, we have to join the labels back again
    predicted_labels = [l[0] + l[1] for l in predicted]
    targets_labels = [l[0] + l[1] for l in targets_test]
    getResults(predicted_labels, targets_labels)

    with open("./test_results.txt", "w") as f:
        for labels, text in zip(zip(predicted_labels, targets_labels), data_test):
            f.write(labels[0] + "\t" + labels[1] + "\t" + text + "\n")

def main():
    train_lines = getTrainFileLines()

    data = np.array([l[1] for l in train_lines])
    targets = np.array([l[0] for l in train_lines])
    logging.debug(targets)

    text_clf = Pipeline([
        ('vect-tfidf', TfidfVectorizer(
            use_idf=True,
            ngram_range=(1, 2),
        )),
        ('clf', MultiOutputClassifier(
            SGDClassifier(
                loss='log_loss',
                penalty=None,
                alpha=0.0001,
                tol=0.001,
                max_iter=50000,
            )
        ))
    ])

    if len(sys.argv) > 1:
        match sys.argv[1]:
            case "optimize":
                optimizeParameters(text_clf, data, targets)
            case "test":
                testOnTrainSet(text_clf, data, targets)
        exit(0)

    # Run on test set
    test_lines = getTestFileLines()

    clf = text_clf.fit(data, targets)
    predicted = clf.predict(test_lines)
    predicted_labels = [l[0] + l[1] for l in predicted]

    # Write results to file
    writeResults(predicted_labels)

if __name__ == "__main__":
    main()
