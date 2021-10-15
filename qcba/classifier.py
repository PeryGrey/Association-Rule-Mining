from sklearn.metrics import accuracy_score
from .qcba_rules import QuantitativeDataFrame


class QuantitativeClassifier:

    def __init__(self, rules, default_class):
        self.rules = rules
        self.default_class = default_class

    def rule_model_accuracy(self, df, actual):
        return accuracy_score(self.predict(df), actual)

    def predict(self, df):
        pred = []

        for _, row in df.dataframe.iterrows():
            found_rule = False
            for rule in self.rules:
                antc_dic = dict(rule.antecedent)
                counter = True

                for name, value in row.iteritems():
                    if name in antc_dic:
                        range_ = antc_dic[name]

                        if type(range_) == str:
                            counter &= range_ == value
                        else:
                            result = range_.isin(value)
                            counter &= result

                if counter:
                    _, pred_class = rule.consequent
                    pred.append(pred_class)
                    found_rule = True
                    break

            if not found_rule:
                pred.append(self.default_class)
        return pred
