#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss, balanced_accuracy_score, confusion_matrix
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone

def make_clf():
    nys = Nystroem(kernel='rbf', n_components=100)
    # logreg w/ default C
    lr = LogisticRegression(class_weight='balanced', C=1., penalty='l2', multi_class='multinomial', max_iter=1000)
    # stitch together
    clf = Pipeline([('nys',nys),('clf',lr)])

    return clone(clf)

def train_classifier(x, y, clf):
    """ train classifier on latent space coordinates

    x: array/dataframe of embedded coordinates
    y: array of labels (n_samples)
    clf: model to train

    returns:
    model: trained model
    """
    ss = StandardScaler()

    # make sure y is set up correctly
    y = np.array(y).astype(int)

    # stitch together preprocessing and model
    model = Pipeline([('scale', ss), ('classify', clf)])

    # train
    model.fit(x, y)

    return model

def return_predictions(train_data, train_labels, test_data):
    """ train model and return predictions and probabilities

    returns:
    train_predictions, train_probs,
    test predictions, test probs
    trained_model
    """
    clf = make_clf()
    model = train_classifier(train_data, train_labels, clf)

    # predict
    train_pred = model.predict(train_data)
    test_pred = model.predict(test_data)
    train_prob = model.predict_proba(train_data)
    test_prob = model.predict_proba(test_data)

    return train_pred, train_prob, test_pred, test_prob, model

def calculate_model_accuracy(train_labels, test_labels, preds, probs):
    """
    calculate model accuracy

    train_labels: array (n_samples), true tissue labels from training data
    test_labels: array (n_samples), true tissue labels from test data
    preds: output from clf.predict
    probs: output from clf.predict_proba

    returns
    accuracy: accuracy for all classes in *both* training and testing data
    logloss: log loss for all classes in *both* training and testing data
    cm: confusion matrix, prediction accuracy for all tissue types in testing data
    """

    intersecting_labels = list(set(np.unique(train_labels)).intersection(set(np.unique(test_labels))))

    accuracy = balanced_accuracy_score(test_labels[np.isin(test_labels, intersecting_labels)],
                    preds[np.isin(test_labels, intersecting_labels)])

    logloss = log_loss(test_labels,
                        probs[:,np.isin(np.unique(train_labels), intersecting_labels)], labels=intersecting_labels)


    # make cm
    cm = np.zeros((len(np.unique(test_labels)), len(np.unique(train_labels))))
    y_labels=[]
    x_labels=[]

    for n,i in enumerate(pd.unique(test_labels)):
        pred_data = preds[test_labels==i]
        total_n = len(pred_data)

        for m,j in enumerate(pd.unique(train_labels)):
            cm[n,m] = np.sum(pred_data==j)

        cm[n,:] = cm[n,:]/total_n

    return accuracy, logloss, cm
