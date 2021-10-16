import pandas
import numpy as np
from ..qcba_rules import QuantitativeDataFrame
from ..range_iterator import Range


class PruneLiterals:

    def __init__(self, quantitative_dataframe):
        self.__dataframe = quantitative_dataframe

    def transform(self, rules):
        return [self.__trim(rule) for rule in rules]

    def __trim(self, rule):
        def transfer_rule(rule, copied_rule):
            # Method for transfering the properties of the rule from candidate to rule
            rule.support = copied_rule.support
            rule.confidence = copied_rule.confidence
            rule.rulelen = copied_rule.rulelen
            rule.antecedent = copied_rule.antecedent

        # Flag to mention if a literal has been removed from the antecdent
        removed = False

        literals, consequent = rule.antecedent, rule.consequent
        rule.update_properties(self.__dataframe)

        # If only one literals are present then the pruning is not done
        if len(literals) < 1:
            return rule

        while True:
            for pos in range(len(literals)):
                
                # Removing 1 rule at a time
                c_literal = literals[0:pos] + literals[pos+1:len(literals)]
                
                c_rule = rule.copy()
                c_rule.antecedent = c_literal

                # updating the confidence of the candidate rule
                c_rule.update_properties(self.__dataframe)

                if c_rule.confidence > rule.confidence:
                    # If there is an improvement in the confidence update the rule
                    transfer_rule(rule, c_rule)
                    removed = True
                    break
                else:
                    removed = False
            
            # if no literal has been removed, the algorithm stops
            if removed == False:
                break
        return rule
