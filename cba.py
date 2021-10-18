from classifiers.m1classifier import M1Classifier
from classifiers.m2classifier import M2Classifier

from rule_generator import convertToCARs, generateARs
from models import TransactionDB


class ClassBasedAssoc:
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
        self.available_algorithms = {"m1": M1Classifier, "m2": M2Classifier}

    def rule_model_accuracy(self, txns):
        if not self.classifier:
            raise Exception("CBA must be trained using fit method first")
        if not isinstance(txns, TransactionDB):
            raise Exception("txns must be of type TransactionDB")

        return self.classifier.test_transactions(txns)

    def generateCARS(self, transactions):
        assoc_rules = None

        assoc_rules = generateARs(  # generate top association rules
            transactions,
            support=self.support,
            confidence=self.confidence,
            maxlen=self.maxlen,
        )

        cars = convertToCARs(assoc_rules)

        return cars


    def buildClassifier(self, cars, transactions): 
        if self.classification_algorithm == "m1":
            classifier = M1Classifier
        else:
            classifier = M2Classifier
        self.classifier = classifier(
            cars, transactions
        ).train()  

        return self.classifier
