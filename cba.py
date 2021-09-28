from classifiers.m1classifier import M1Classifier
from classifiers.m2classifier import M2Classifier
from rule_algorithm import *
from rule_generator import convertToCARs, generateARs
from models import TransactionDB


class CBA:

    def __init__(self, support=10, confidence=50, maxlen=100, classifier="m1"):
        self.support = support
        self.confidence = confidence
        self.classification_algorithm = classifier
        print(
            "Using support=",
            self.support,
            "conf=",
            self.confidence,
            "classAlgo:",
            self.classification_algorithm,
        )
        self.maxlen = maxlen
        self.classifier = None
        self.target_class = None

    
    def fit(self, transactions):  # transactions = txns_train
        self.target_class = transactions.header[-1]  #'class'
        if self.classification_algorithm == "m1":
            classifier = M1Classifier
        else:
            classifier = M2Classifier

        assoc_rules = None

        assoc_rules = generateARs(  # generate top association rules
            transactions,
            support=self.support,
            confidence=self.confidence,
            maxlen=self.maxlen,
        )

        cars = convertToCARs(assoc_rules)  # covert assoc rules to cars

        self.classifier = classifier(
            cars, transactions
        ).train()  # M1Classifier(cars, txns_train).build() - get the classifier and the rules it is using

        return self.classifier
