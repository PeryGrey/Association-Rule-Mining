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

                # find the data points for the rules that are covered
                current_attribute_values = self.__dataframe.column(attribute)

                # in refit, the rule finds the max and minimum value for the literal.
                refitted_interval = interval.refit(current_attribute_values)

                # updates the interval for the literal
                rule.antecedent[idx] = attribute, refitted_interval

        return rule
