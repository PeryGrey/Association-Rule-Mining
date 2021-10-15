# from .models import QuantitativeCAR
from .transformation import QCBATransformation
from .classifier import QuantitativeClassifier
from .quant_rule import QuantitativeCAR


class QCBA:
    def __init__(self, dataset, cba_rule_model=None):
        self.dataset = dataset
        self.__rules = [QuantitativeCAR(r) for r in cba_rule_model.clf.rules]

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
