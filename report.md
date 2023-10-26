# Truthful vs Deceptive Hotel Reviews - Report

Group 45 - Filipe Silva (95574), Tiago Martins (95678)

## Models

### TD-IDF: Term Frequency times Inverse Document Frequency

The *TF-IDF* statistic is able to capture how important a word is using
statistical techniques. The *TF* (term frequency) part defines the importance of
a word by how frequent it is. This is done by counting the number of times a
word appears in a document, and then dividing it by the total number of words
present in the respective document. The *IDF* (inverse document frequency) part
applies a similar concept as *TF* but at the document level. This metric will
count in how many documents a word appears in and divide it by the total number
of documents. However, it will apply the inverse function to this frequency to
assign lower weights to more frequent words (e.g. "the", "a") that don't add
much value to distinguishing between document labels, and assigns higher weights
to words that seldom appear.
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
results obtained (through splitting of the training data provided into a
training set and a test set; in a 80/20 partition).

Through this parameter optimization, we found that the best results appeared not
for a support vector machine, but for a logistic regression classifier.

After some more testing, we thought about splitting this problem into two:
classification of *POSITIVE*/*NEGATIVE* and *TRUTHFUL*/*DECEPTIVE*, instead of
the original 4 classes. This way, we might end up with a somewhat better
accuracy after the two models' predictions are joined. Alas, we used a
`MultiOutputClassifier` which used the logistic regression for both problems.
Due to a lack of space we won't show the results here, but the average accuracy
over multiple runs improved.
<!---
 TODO get some results
-->

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

To classify reviews, our two models will train over the provided training
dataset and distinguish between our two sets of classes: *POSITIVE*/*NEGATIVE*
and *TRUTHFUL*/*DECEPTIVE*.

We can then feed a review to our model and it'll provide us with a proper 
classification of that review, according to what it thinks the most fitting 
label is.

Keeping in mind that the current *metrics* package of *scikit-learn* doesn't
support multi output classification, we joined the results at the end to take
advantage of the model reports.

### Training/Testing

We trained our model using the parameters described above, over the provided
training file that contains 1400 reviews that have already been labeled. As a
first step to evaluate our model, we split the training data into a train set
and into a test set using a 80/20 ratio, as stated before. The precision,
recall, f1-score and overall accuracy were taken directly from the test set.
Below, its possible to see a table representing the first 3 metrics. The
accuracy obtained was 84%.

| Label             | Precision | Recall | F1-score |
| ----------------- | --------- | ------ | ---------|
| DECEPTIVENEGATIVE |    0.84   |  0.82  |   0.83   |
| DECEPTIVEPOSITIVE |    0.83   |  0.91  |   0.87   |
| TRUTHFULNEGATIVE  |    0.80   |  0.76  |   0.78   |
| TRUTHFULPOSITIVE  |    0.87   |  0.85  |   0.86   |

Its also possible to inspect the confusion matrix generated from our model for 
a deeper insight on each label:

| Predicted\\Actual | DN | DP | TN | TP |
| ----------------- | -- | -- | -- | -- |
| DN                | 56 |  3 |  9 |  0 |
| DP                |  2 | 58 |  0 |  4 |
| TN                |  8 |  0 | 48 |  7 |
| TP                |  1 |  9 |  3 | 72 |

### Discussion

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