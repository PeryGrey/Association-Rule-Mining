import pandas
import numpy as np
from ..qcba_rules import QuantitativeDataFrame
from ..range_iterator import Range


class Refit:
    def __init__(self, dataframe):
        self.__dataframe = dataframe

    def transform(self, rules):
        return [self.__refit(r) for r in rules]

    def __refit(self, rule):
        for idx, literal in enumerate(rule.antecedent):
            attribute, interval = literal

            if type(interval) != str:
                current_attribute_values = self.__dataframe.column(attribute)
                refitted_interval = interval.refit(current_attribute_values)
                rule.antecedent[idx] = attribute, refitted_interval

        return rule
