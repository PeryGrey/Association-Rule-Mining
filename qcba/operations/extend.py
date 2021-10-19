import pandas
import numpy as np
import math
from tqdm import tqdm

from ..qcba_rules import QuantitativeDataFrame
from ..range_iterator import Range


class Extend:

    def __init__(self, dataframe):
        self.__dataframe = dataframe

    def transform(self, rules):
        copied_rules = [rule.copy() for rule in rules]

        extended_rules = []
        for i in tqdm(rules):
            extended_rules.append(self.__extend_rule(i))
        return extended_rules

    def __extend_rule(self, rule, min_improvement=0, min_conditional_improvement=-0.01):
        current_best = rule
        direct_extensions = self.__get_extensions(rule)

        current_best.update_properties(self.__dataframe)

        while True:
            extension_succesful = False

            # Get the candidates after extending the rules
            direct_extensions = self.__get_extensions(current_best)

            for candidate in direct_extensions:
                candidate.update_properties(self.__dataframe)

                # checks for the improvement
                delta_confidence = candidate.confidence - current_best.confidence
                delta_support = candidate.support - current_best.support

                # if the confidence is above min_improvement and the rule have increased in support
                if delta_confidence >= min_improvement and delta_support > 0:
                    current_best = candidate
                    extension_succesful = True
                    break

                if delta_confidence >= min_conditional_improvement:
                    enlargement = candidate

                    while True:
                        # if the cand rule does not have enough support, it tries to extend the literal as much as possible 
                        # until there is no possibility of any extension.
                        enlargement = self.get_beam_extensions(enlargement)

                        if enlargement is None:
                            break

                        candidate.update_properties(self.__dataframe)
                        enlargement.update_properties(self.__dataframe)

                        delta_confidence = enlargement.confidence - current_best.confidence
                        delta_support = enlargement.support - current_best.support

                        if delta_confidence >= min_improvement and delta_support > 0:
                            current_best = enlargement
                            extension_succesful = True

                        elif delta_confidence >= min_conditional_improvement:
                            continue
                        else:
                            break
                    if extension_succesful == True:
                        break
                else:
                    continue
            if extension_succesful == False:
                break
        return current_best

    def __get_extensions(self, rule):
        def inner_loop(literal, extended_literal):
            copied_rule = rule.copy()
            current_literal_index = copied_rule.antecedent.index(literal)
            copied_rule.antecedent[current_literal_index] = extended_literal
            copied_rule.was_extended = True
            copied_rule.extended_literal = extended_literal
            return copied_rule

        extended_rules = []

        for literal in rule.antecedent:
            attribute, interval = literal
            neighborhood = self.__get_direct_extensions(literal)

            for extended_literal in neighborhood:
                copied_rule = inner_loop(literal, extended_literal)
                extended_rules.append(copied_rule)

        extended_rules.sort(reverse=True)

        return extended_rules

    def __get_direct_extensions(self, literal):
        attribute, interval = literal
        if type(interval) == str:
            return [literal]

        vals = self.__dataframe.column(attribute)
        mask = interval.test_membership(vals)
        member_indexes = np.where(mask)[0]

        first_index_modified = member_indexes[0] - 1
        last_index_modified = member_indexes[-1] + 1

        new_left_bound = interval.minimumvalue
        new_right_bound = interval.maximumvalue

        # prepare return values
        extensions = []

        if not first_index_modified < 0:
            new_left_bound = vals[first_index_modified]
            temp_interval = Range(
                new_left_bound, interval.maximumvalue,  True, interval.right_boundary)
            extensions.append((attribute, temp_interval))

        if not last_index_modified > len(vals) - 1:
            new_right_bound = vals[last_index_modified]
            temp_interval = Range(
                interval.minimumvalue, new_right_bound, interval.left_boundary, True)
            extensions.append((attribute, temp_interval))

        return extensions

    def get_beam_extensions(self, rule):
        if not rule.was_extended:
            print(rule.was_extended)
            return None

        literal = rule.extended_literal
        extended_literal = self.__get_direct_extensions(literal)

        if len(extended_literal) == 0:
            return None

        copied_rule = rule.copy()
        literal_index = copied_rule.antecedent.index(literal)

        copied_rule.antecedent[literal_index] = extended_literal[0]
        copied_rule.was_extended = True
        copied_rule.extended_literal = extended_literal[0]

        return copied_rule
