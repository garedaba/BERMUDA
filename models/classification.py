#!/usr/bin/env python
# coding: utf-8
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss, balanced_accuracy_score, confusion_matrix

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


def calculate_model_accuracy(true, predicted):
    """Calculate log-loss and balanced accuracies of multiclass prediction

    true: class labels, n_samples
    predicted: output from model.predict_proba

    returns:
    logistic_loss, accuracy: normalsied log_loss and adjusted balanced accuracy (0=random)
    confusion: confusion matric
    """
    hard_class = np.argmax(predicted, axis=1)

    accuracy = balanced_accuracy_score(true, hard_class, adjusted=True)
    logistic_loss = log_loss(true, predicted)

    confusion = confusion_matrix(true, hard_class)

    return logistic_loss, accuracy, confusion
