import logging
import sys

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import ClassifierChain
from sklearn import metrics

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='===== %(levelname)s[%(lineno)d] =====\n%(message)s\n'
)

def getHonesty(string):
    return 1 if "TRUTHFUL" in string else 0


def getPolarity(string):
    return 1 if "POSITIVE" in string else 0


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
        # 'vect-tfidf__use_idf': (True, False),
        # 'vect-tfidf__ngram_range': [(1, 1), (1, 2), (1, 3), (1, 4)],
        'clf__loss': ('hinge', 'log_loss', 'modified_huber', 'squared_hinge', 'perceptron'),
        'clf__penalty': ('l1', 'l2', 'elasticnet', None),
        'clf__alpha': (1e-2, 1e-3, 1e-4, 1e-5, 1e-6),
        'clf__tol': (None, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6),
    }

    gs_clf = GridSearchCV(model, parameters, cv=5, n_jobs=-2)
    gs_clf = gs_clf.fit(data, targets)
    for param_name in sorted(parameters.keys()):
        logging.info("%s: %r" % (param_name, gs_clf.best_params_[param_name]))

    exit(0)


def main():
    train_lines = getTrainFileLines()
    test_lines = getTestFileLines()

    data = [l[1] for l in train_lines]
    targets = [l[0] for l in train_lines]
    logging.debug(targets)

    text_clf = Pipeline([
        ('vect-tfidf', TfidfVectorizer(
            use_idf=True,
            ngram_range=(1, 2),
        )),
        ('clf', ClassifierChain(
            SGDClassifier(
                loss='log_loss',
                penalty=None,
                alpha=0.0001,
                tol=0.001,
                max_iter=10000,
                # random_state=42, # For reproducibility
            )
        ))
    ])

    data_train, data_test, targets_train, targets_test = train_test_split(
        data,
        targets,
        test_size=0.25,
        # random_state=42 # For reproducibility TODO might remove?
    )

    if len(sys.argv) > 1 and sys.argv[1] == "train":
        optimizeParameters(text_clf, data, targets)

    clf = text_clf.fit(data_train, targets_train)
    predicted = clf.predict(data_test)

    # The metrics package does not support multioutput, so we have to do it manually
    t1 = []
    for el in predicted:
        t1.append(str(int(el[0])) + str(int(el[1])))

    t2 = []
    for el in targets_test:
        t2.append(str(el[0]) + str(el[1]))

    # Print the classification report
    logging.info(metrics.classification_report(t1, t2))
    logging.info(metrics.classification_report(t1, t2, output_dict=True)['accuracy'])

    # Print and plot the confusion matrix
    cm = metrics.confusion_matrix(t1, t2)
    logging.info(cm)

    # Write results to file TODO uncomment
    # writeResults(clf.predict(test_lines))

if __name__ == "__main__":
    main()
