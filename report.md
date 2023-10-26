# Truthful vs Deceptive Hotel Reviews - Report

Group 45 - Filipe Silva (95574), Tiago Martins (95678)

## Models

### TD-IDF: Term Frequency times Inverse Document Frequency

The *TF-IDF* statistic is able to capture how important a word is using
statistical techniques. It consists of two parts:

* The *TF* (term frequency) part defines the importance of a word by how
  frequent it is in a document;
* The *IDF* (inverse document frequency) part applies a similar concept as *TF*
  but at the document level ($ln({number\_of\_documents \over
  number\_of\_documents\_where\_the\_word\_appears})$)
  Apart from the natural logarithm, the formula is inverted, so that more
  frequent words (e.g. "the", "a") which don't help us distinguish between
  document labels have lower weight, and words that seldom appear have higher
  weight.

*TF-IDF* multiplies both these metrics to obtain a score for each word that will
reflect its importance. The higher the score is, the more important the word is.

### SVMs: Support Vector Machines

SVMs are one of the most popular methods to perform classification of data. In
this case, it can also be very useful to classify text. The main objective of
SVMs is to find a hyperplane that better divides the data space into its
respective classes. The better defined this hyperplane is, the better
classification it'll make.

### Logistic regression

Logistic regression is a type of probabilistic classification, which means that
it predicts a probability distribution over a set of classes, instead of only
the most likely class for a given observation.

## Experimental Setup and Results

### Iteration through models

At first, we created a model following a [scikit-learn
tutorial](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html#loading-the-20-newsgroups-dataset),
that used a `CountVectorizer` to do text processing, tokenizing and filtering of
stopwords; a `TdidfTransformer` to calculate frequencies based on the occurrence
of the words; and finally a `SGDCClassifier` implementing a support vector
machine to predict the label for a given observation. It also allowed for
parameter tuning of the different steps of the pipeline, and metrics for the
results obtained. Through this parameter optimization, we found that the best
results appeared not for a support vector machine, but for a logistic regression
classifier.

After some more testing, we thought about splitting this problem into two:
classification of *POSITIVE*/*NEGATIVE* and *TRUTHFUL*/*DECEPTIVE*, instead of
the original 4 classes. This way, we might end up with a somewhat better
accuracy after the two models' predictions are joined. Alas, we used a
`MultiOutputClassifier` which used the logistic regression for both problems.
Over 20 runs, the average accuracy of the approaches using the old model and the
new one was 0.826 and 0.832, respectively. In spite of the increase in accuracy
being small, we found this approach interesting and decided to keep using it.
As the current *metrics* package of *scikit-learn* doesn't support multi output
classification, we joined the results at the end to take advantage of the model
reports.

### Final model

We ended up with the following parameters for the different steps of
the pipeline:

* Vectorizer/TF-IDF:
  * *ngram_range*: (1, 2) [extract both unigrams and bigrams]

* SGDCClassifier:
  * *loss*: 'log_loss' [use logistic regression instead of a SVM]
  * *penalty*: None [no penalty is added for regularization]
  * *alpha*: 0.0001 [used to compute the learning rate, multiplier for the
    penalty]
  * *tol*: 0.001 [stopping criterion; training will stop once the loss is
    greater than the best loss - *tol*]

## Training/Testing

We trained our model using the parameters described above, over the provided
training file. In order evaluate our model, we split the training data
into a train set and into a test set using a 80/20 ratio.
Below, it's possible to see a table displaying the precision, recall and
F1-score for each one of the labels, in one of the tests. The accuracy obtained
was 86.4%.

| Label               | Precision | Recall | F1-score |
| ------------------- | --------- | ------ | ---------|
| *DECEPTIVENEGATIVE* |    0.87   |  0.87  |   0.87   |
| *DECEPTIVEPOSITIVE* |    0.96   |  0.82  |   0.89   |
| *TRUTHFULNEGATIVE*  |    0.77   |  0.85  |   0.81   |
| *TRUTHFULPOSITIVE*  |    0.82   |  0.88  |   0.85   |

Its also possible to inspect the confusion matrix generated from our model for 
a deeper insight on each label: (T: TRUTHFUL, D: DECEPTIVE, P: POSITIVE, N:
NEGATIVE)

| True\\Predicted  | DN | DP | TN | TP | Total |
| ---------------- | -- | -- | -- | -- | ----- |
| *DN*             | 62 |  0 |  9 |  0 | 71    |
| *DP*             |  4 | 66 |  1 |  9 | 80    |
| *TN*             |  5 |  0 | 51 |  4 | 60    |
| *TP*             |  0 |  3 |  5 | 61 | 69    |
| Total            | 71 | 69 | 66 | 74 | 280   |

### Discussion

Looking at the results for the run shown above, we found some errors: (line
numbers are for the file `train.txt`)

* Inability to detect if a certain event is likely or not
  * In line 85, our model reported it as *TRUTHFULNEGATIVE* where the correct
    labeling is *DECEPTIVENEGATIVE*. We believe this is due to the last sentence
    in particular: 200$ of extra charges is not something common, and the model
    is not aware of that.
  * In line 888, our model did the same thing as above, not considering that the
    probability of being checked in into a room where someone else is already
    staying is really low.

* Hyper focus on certain parts of the review
  * For the review in line 295, our model reported it as *TRUTHFULPOSITIVE* and
    the correct labeling was *TRUTHFULNEGATIVE*. We believe that the model
    assigned more importance to the positive points shown at the end of the
    review, disregarding most the of middle portion which showed the negative
    points.
  * On line 200, our model reported *DECEPTIVENEGATIVE* but it should be
    *DECEPTIVEPOSITIVE*. We believe that was due to the toilet incident reported
    in the review. It correctly detected that the review was deceptive (the
    reference to "gold bathroom" might have helped).

* Complete hallucinations
  * On line 1188, our model reported *TRUTHFULNEGATIVE* when it should be
    *DECEPTIVEPOSITIVE* (the only time both the polarity and truthfulness were
    both wrongly predicted by our model in this run). We do not have any
    explanation for this one, it is completely wrong.

We also found some contentious review labellings:

* Line 522: (our model reported *TRUTHFULNEGATIVE* while it should've been
  *TRUTHFULPOSITIVE*) the ending had a good connotation to it, but the situation
  reported in the beginning could warrant a negative labelling.
* Line 78: (our model reported *TRUTHFULPOSITIVE* while it should've been
  *TRUTHFULNEGATIVE*) due to one bad experience that could have nothing to do
  with the hotel, all the good things are disregarded.

All in all, in this run our model failed 38 labellings out of 280 (86.4%
accuracy), which means that notwithstanding these failures depicted above, the
model is quite accurate.

### Future work

Due to time constraints, we could not implement any extra features into this
classifier. However, here we describe some of the ideas we had for further
improvements, or just some interesting approaches that we did not have time to
pursue:

* Train and use a neural network to perform classification;
* Use GPT-3.5 or GPT-4 and "ask" it for a label given an observation and a list
  of observations with their correspondent labels. This [GitHub
  repository](https://github.com/xtekky/gpt4free) seems interesting for this
  purpose;
* Use 3 different models (e.g. the classifier we used, a neural network and GPT)
  and run the same task of classification on all of them, and then choose the
  label with the most "votes" per observation (this could improve accuracy by
  complementing one model with another);
* Try a RegEx-based classifier (it probably wouldn't be as resilient to change,
  or as easy to implement as the approaches above, but it would be interesting
  to see what results we could achieve).