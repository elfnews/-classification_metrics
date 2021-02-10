# Import our libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC


def accuracy(actual, predictions):
    """
    INPUT
    :param actual: actual values as a numpy array or pandas series
    :param predictions: predictions as a numpy array or pandas series

    :return: returns the accuracy as a float
    """
    return np.count_nonzero(predictions == actual) / len(actual)


# precision is the true positives over the predicted positive values
def precision(actual, predictions):
    """
    INPUT
    (assumes positive = 1 and negative = 0)
    :param actual: actual values as a numpy array or pandas series
    :param predictions: predictions as a numpy array or pandas series

    :return: returns the precision as a float
    """
    tp = len(np.intersect1d(np.where(predictions == 1), np.where(actual == 1)))
    predictions_positives = np.count_nonzero(predictions == 1)
    return tp / predictions_positives


# recall is true positives over all actual positive values
def recall(actual, predictions):
    """
    INPUT
    :param actual: actual values as a numpy array or pandas series
    :param predictions: predictions as a numpy array or pandas series

    :return: returns the recall as a float
    """
    tp = len(np.intersect1d(np.where(predictions == 1), np.where(actual == 1)))
    actual_positives = np.count_nonzero(actual == 1)
    return tp / actual_positives


# f1_score is 2*(precision*recall)/(precision+recall)
def f1(actual, predictions):
    """
    INPUT
    :param actual: actual values as a numpy array or pandas series
    :param predictions: predictions as a numpy array or pandas series
    :return: returns the f1score as a float
    """
    precision1 = precision(actual=actual, predictions=predictions)
    recall1 = recall(actual=actual, predictions=predictions)
    return 2 * precision1 * recall1 / (precision1 + recall1)


# Read our dataset
df = pd.read_table('smsspamcollection/SMSSpamCollection',
                   sep='\t',
                   header=None,
                   names=['label', 'sms_message'])

# Fix our response value
df['label'] = df.label.map({'ham': 0, 'spam': 1})

# Split our dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(df['sms_message'],
                                                    df['label'],
                                                    random_state=1)

# Instantiate teh CountVectorizer method
count_vector = CountVectorizer()

# Fist the training data and then return the matrix
training_data = count_vector.fit_transform(X_train)

# Transform testing data and return the matrix. Note we are not fitting the testing data into the CountVectorizer()
testing_data = count_vector.transform(X_test)

# Instantiate a number of our models
naive_bayes = MultinomialNB()
bag_mod = BaggingClassifier(n_estimators=200)
rf_mod = RandomForestClassifier(n_estimators=200)
ada_mod = AdaBoostClassifier(n_estimators=300, learning_rate=0.2)
svm_mod = SVC()

naive_bayes.fit(training_data, y_train)
bag_mod.fit(training_data, y_train)
rf_mod.fit(training_data, y_train)
ada_mod.fit(training_data, y_train)
svm_mod.fit(training_data, y_train)

# The models you fit aboive were fit on which data?
a = 'X_train'
b = 'X_test'
c = 'y_train'
d = 'y_test'
e = 'training_data'
f = 'testing_data'

# Change models_fit_on to only contain the correct string names
# of values that you passed to the above models

models_fit_on = {c, e}  # update this to only contain correct letters

# Checks your solution - don't change this
# t.test_one(models_fit_on)

# Make predictions using each of your models
predictions_nb = naive_bayes.predict(testing_data)
predictions_bag = bag_mod.predict(testing_data)
predictions_rf = rf_mod.predict(testing_data)
predictions_ada = ada_mod.predict(testing_data)
predictions_svm = svm_mod.predict(testing_data)

print('Accuracy:')
print(accuracy(actual=y_test, predictions=predictions_nb))
print(accuracy_score(y_true=y_test, y_pred=predictions_nb))

print('Precision:')
print(precision(actual=y_test, predictions=predictions_nb))
print(precision_score(y_true=y_test, y_pred=predictions_nb))

print('Recall:')
print(recall(actual=y_test, predictions=predictions_nb))
print(recall_score(y_true=y_test, y_pred=predictions_nb))

print('F1:')
print(f1(actual=y_test, predictions=predictions_nb))
print(f1_score(y_true=y_test, y_pred=predictions_nb))

print('FBeta:')
print(fbeta_score(y_true=y_test, y_pred=predictions_nb, beta=1))
