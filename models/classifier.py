import pandas as pd
from functools import reduce

from models import ClassAssocationRule, Antecedent, Consequent, TransactionDB


class Classifier:
    def __init__(self):
        self.rules = []
        self.default_class = None
        self.default_class_attribute = None
        self.default_class_confidence = None
        self.default_class_support = None

        self.default_rule = None

    def rule_model_accuracy(self, txns):
        if not self.clf:
            raise Exception("CBA must be trained using fit method first")
        if not isinstance(txns, TransactionDB):
            raise Exception("txns must be of type TransactionDB")

        return self.clf.test_transactions(txns)

    def test_transactions(self, txns):
        pred = self.predict_all(txns)
        actual = txns.classes

        return accuracy_score(pred, actual)

    def predict(self, datacase):
        for rule in self.rules:
            if rule.antecedent <= datacase:
                return rule.consequent.value

        return self.default_class

    def predict_all(self, dataset):
        predicted = []

        for datacase in dataset:
            predicted.append(self.predict(datacase))

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
