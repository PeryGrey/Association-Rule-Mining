import pandas
import numpy as np
from ..qcba_rules import QuantitativeDataFrame, QuantitativeCAR
from ..range_iterator import Range


class Trim:
    def __init__(self, quantitative_dataframe):
        self.__dataframe = quantitative_dataframe

    def transform(self, rules):
        r = [rule.copy() for rule in rules]
        return [self.__trim(rule) for rule in r]

    def __trim(self, rule):
        antecedent_mask, consequent_mask = self.__dataframe.find_covered_by_rule_mask(
            rule)
        covered_by_rule_mask = antecedent_mask & consequent_mask

        # instances covered by rule
        covered_by_r = self.__dataframe.mask(covered_by_rule_mask)

        antecedent = rule.antecedent

        for idx, literal in enumerate(antecedent):
            attribute, interval = literal
            if type(interval) == str:
                continue

            current_column = covered_by_r[[attribute]].values
            if not np.unique(current_column).any():
                continue

            minv = min(current_column)[0]
            maxv = max(current_column)[0]

            new_interval = Range(minv, maxv, True, True)
            antecedent[idx] = attribute, new_interval

        return rule
