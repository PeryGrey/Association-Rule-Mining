from cba import ClassBasedAssoc
from models import TransactionDB
import pandas as pd
from sklearn.utils import shuffle
from qcba import (
    RangeIterator,
    Range,
    QuantitativeDataFrame,
    QuantitativeCAR,
    QCBATransformation,
    QuantitativeClassifier
)


class QCBA:
    def __init__(self, dataset, cba_rule_model=None):
        self.dataset = dataset
        self.__rules = [QuantitativeCAR(r)
                        for r in cba_rule_model.classifier.rules]

        self.transformation = QCBATransformation(dataset)
        self.clf = None

    def fit(self, stages):
        transformed_rules, default_class = self.transformation.transform(
            self.__rules, stages)
        self.clf = QuantitativeClassifier(transformed_rules, default_class)
        return self.clf

    def score(self, dataset):
        actual = dataset.dataframe.iloc[:, -1]
        return self.clf.rule_model_accuracy(dataset, actual)


empty_string = ""
null_string = "NULL"
inf_string = "inf"

range_iterator = RangeIterator()

range_iterator.closed_bracket = empty_string, null_string
range_iterator.open_bracket = null_string, empty_string
range_iterator.infinity_symbol = inf_string, inf_string
range_iterator.members_separator = "_to_"

range_iterator.initialize_reader()

QuantitativeCAR.range_iterator = range_iterator

path = 'iris'

data_train_discretized = pd.read_csv(
    f"datasets/qcba-datasets/binned-{path}.csv")
data_train_undiscretized = pd.read_csv(f"datasets/qcba-datasets/{path}.csv")
data_test = pd.read_csv(f"datasets/qcba-datasets/binned-{path}.csv")

txns_train = TransactionDB.from_DataFrame(data_train_discretized)
txns_test = TransactionDB.from_DataFrame(data_test)

quant_dataframe_train_disc = QuantitativeDataFrame(data_train_discretized)
quant_dataframe_train_undisc = QuantitativeDataFrame(data_train_undiscretized)

cba = ClassBasedAssoc()
cars = cba.generateCARS(txns_train)
classifier = cba.buildClassifier(cars, txns_train)
cba.rule_model_accuracy(txns_train)

print("-"*50)

qcba_cba = QCBA(quant_dataframe_train_undisc, cba_rule_model=cba)
qcba_stages = {
    "refitting": True,
    "literal_pruning": True,
    "trimming": True,
    "extension": True,
    "overlap_pruning": True,
    "based_drop": True}

qcba_cba.fit(qcba_stages)

print("-"*50)
print("CBA accuracy:", cba.rule_model_accuracy(txns_train))
print("QCBA accuracy:", qcba_cba.score(quant_dataframe_train_undisc))
