import pandas
import numpy as np

from ..data_structures import QuantitativeDataFrame, Interval


class Refit:
    def __init__(self, dataframe):
        self.__dataframe = dataframe
        
    def transform(self, rules):
        r = [rule.copy() for rule in rules]
        refitted = [self.__refit(rule) for rule in r]
        return refitted
        
    def __refit(self, rule):
        for idx, literal in enumerate(rule.antecedent):
            attribute, interval = literal
            if type(interval) != str:
                current_attribute_values = self.__dataframe.column(attribute)
                refitted_interval = interval.refit(current_attribute_values)
                rule.antecedent[idx] = attribute, refitted_interval

        return rule
            