# Truthful vs Deceptive Hotel Reviews - Report

Group 45 - Filipe Silva (95574), Tiago Martins (95678)

## Models

### TD-IDF: Term Frequency times Inverse Document Frequency

The *TF-IDF* statistic is able to capture how important a word is by using
simple statistical techniques to assign such importance. There are two main
parts to this model: *TF* and *IDF*. The *TF* (term frequency) defines the
importance of a word by how frequent it is. This is trivially done by counting
the number of times a word appears in a document. The importance can then be
expressed in probabilistic terms, by diving the frequency value by the total
number of words present in the respective document. The *IDF* (inverse document
frequency) applies a similar concept as *TF* but at the document level. This
metric will count in how many documents a word appears in and divide it by the
total number of documents. However, it will apply the inverse function to this
frequency to assign lower weights to more frequent words like "the" or "a" that
don't add much value to distinguishing between document labels, and assigns
higher weights to words that seldom appear. *TF-IDF* will then multiply both
these metrics to obtain a score for each word that will reflect its importance.
The higher the score is, the more important the word is. *TF-IDF* is used in our
model as a preprocessing step to convert words into numbers so that they can be
used by our model. This is an important step as it will server as the
cornerstone for the rest of our model.


### Falar do "optimizeParameters"?


### Support Vector Machines

SVMs are one of the most popular methods to perform classification of data. In
this case, it can also be very useful to classify text. The main objective of
SVMs is to find a hyperplane that better divides the data space into its
respective classes. The better defined this hyperplane is, the better
classification it'll make.

To classify reviews, our model will train over the provided training dataset 
and define a hyperplane to distinguish between our classes: 

* *TRUTHFULPOSITIVE*
* *TRUTHFULNEGATIVE*
* *DECEPTIVEPOSITIVE*
* *DECEPTIVENEGATIVE*

We can then feed a review to our model and it'll provide us with a proper 
classification of that review, according to what it thinks the most fitting 
label is.
Our model has a base parameterization as follows:

* *penalty*: elasticnet. It combines the regularization techniques known as L1
  and L2 to provide a balance between them in an effort to provide a more
  efficient regularization technique;
* *alpha*: 0.0001. This value gives a weight to the penalty applied during
  training;
* *tol*: None. Indicates that training will stop once the loss if greater than
  the best loss over a predefined number of consecutive rounds;
* *max_iter*: 1000. The model will train up to a maximum number of iterations.

## Experimental Setup and Results

We trained our model using the parameters described above, over the provided
training file that contains 1400 reviews that have already been labeled. As a
first step to evaluate our model, we split the training data into a train set
and into a test set using a 80/20 ratio, respectively. The precision, recall,
f1-score and overall accuracy were taken directly from the test set. Below, its
possible to see a table representing the first 3 metrics. The accuracy obtained
was 84%.

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