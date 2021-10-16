import pandas
import numpy as np
from ..qcba_rules import QuantitativeDataFrame, QuantitativeCAR
from ..range_iterator import Range


class Trim:
    def __init__(self, quantitative_dataframe):
        self.__dataframe = quantitative_dataframe

    def transform(self, rules):
        return [self.__trim(rule) for rule in rules]

    def __trim(self, rule):
        ant_cover, consq_cover = self.__dataframe.find_covered_by_rule_mask(
            rule)
        rule_cover = ant_cover & consq_cover

        # instances covered by rule
        rule_mask = self.__dataframe.mask(rule_cover)
        antc = rule.antecedent

        for idx, literal in enumerate(antc):
            attribute, interval = literal

            # if the literal was originally of numerical type
            if type(interval) != str:

                current_column = rule_mask[attribute].values    
                if len(current_column) == 0:
                    continue

                minv = min(current_column)
                maxv = max(current_column)

                new_interval = Range(minv, maxv, True, True)
                antc[idx] = attribute, new_interval

        return rule
