import pandas
import numpy as np
from ..qcba_rules import QuantitativeDataFrame
from ..range_iterator import Range


class Prune_Overlap:

    def __init__(self, quantitative_dataset):
        self.__dataframe = quantitative_dataset

    def copy_rules(self, rules):
        return [rule.copy() for rule in rules]

    def transform(self, rules, default_class, transaction_based=True):
        copied_rules = self.copy_rules(rules)

        pruned_rules = copied_rules
        pruned_rules = self.prune_transaction_based(
            copied_rules, default_class)

        return pruned_rules

    def prune_transaction_based(self, rules, default_class):
        new_rules = self.copy_rules(rules)

        for idx, rule in enumerate(rules):            
            rule_classname, rule_classval = rule.consequent
            
            # Iterates over all the rules to check if the rules class is same as the default class
            if rule_classval != default_class:
                continue

            cca, ccv = self.__dataframe.find_covered_by_rule_mask(rule)
            correctly_covered = cca & ccv

            flag = False

            for candidate_clash in rules[idx:]:
                cand_classname, cand_classval = candidate_clash.consequent

                # removes the rule if it is covered by default class
                if cand_classval == default_class:
                    continue

                cand_clash_covered_antecedent, _ = self.__dataframe.find_covered_by_rule_mask(
                    candidate_clash)
                if any(cand_clash_covered_antecedent & correctly_covered):
                    flag = True
                    break

            if flag == False:
                new_rules.remove(rule)

        return new_rules
