import pandas
import numpy as np

from ..data_structures import QuantitativeDataFrame, Interval


class PruneLiterals:
    
    def __init__(self, quantitative_dataframe):
        self.__dataframe = quantitative_dataframe
        
    def transform(self, rules):
        r = [rule.copy() for rule in rules]        
        return [ self.__trim(rule) for rule in r ]
    
    def __trim(self, rule):
        def transfer_rule(rule, copied_rule):
            rule.support = copied_rule.support
            rule.confidence = copied_rule.confidence
            rule.rulelen = copied_rule.rulelen
            rule.antecedent = copied_rule.antecedent

        removed = False
   
        literals = rule.antecedent
        consequent = rule.consequent
        
        rule.update_properties(self.__dataframe)
        
        dataset_len = self.__dataframe.size

        if len(literals) < 1:
            return rule

        while True:
            for pos in range(len(literals)):
                literals_combination = literals[0:pos] + literals[pos+1:len(literals)]
                
                c_rule = rule.copy()
                c_rule.antecedent = literals_combination
                c_rule.update_properties(self.__dataframe)

                if c_rule.confidence > rule.confidence:
                    transfer_rule(rule, c_rule)
                    removed = True
                    break                    
                else:
                    removed = False
            if removed == False:
                break
        return rule