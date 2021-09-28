import pandas as pd
from functools import reduce

from models import ClassAssocationRule, Antecedent, Consequent

#Evaluate a classifier against a given dataset
def evaluate(classifier, txns):
    pred = predict_all(classifier, txns)
    actual = txns.classes

    return accuracy_score(pred, actual)


# predict for single data value
def predict(classifier, datacase):
    for rule in classifier.rules:
        if rule.antecedent <= datacase:
            return rule.consequent.value

    return classifier.default_class


# predict for entire dataset
def predict_all(classifier, dataset):
    predicted = []

    for datacase in dataset:
        predicted.append(predict(classifier, datacase))

    return predicted


def accuracy_score(actual, predicted):
    length = len(actual)

    indices = range(length)

    def reduce_indices(previous, current):
        i = current

        result = 1 if actual[i] == predicted[i] else 0

        return previous + result

    accuracy = reduce(reduce_indices, indices) / length

    return accuracy
