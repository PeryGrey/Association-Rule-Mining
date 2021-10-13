import pandas
import numpy as np

from ..data_structures import QuantitativeDataFrame, Interval


class Prune_Overlap:
    
    def __init__(self, quantitative_dataset):
        self.__dataframe = quantitative_dataset
        
        
    def transform(self, rules, default_class, transaction_based=True):
        return self.prune_transaction_based([ i.copy() for i in rules ], default_class)
    
    def prune_transaction_based(self, rules, default_class):
        new_rules = [ i for i in rules ]
        for idx, rule in enumerate(rules):            
            rule_classname, rule_classval = rule.consequent
            if rule_classval != default_class:
                continue

            cor_cover_antc, cor_cover_consq = self.__dataframe.find_covered_by_rule_mask(rule)
            cor_cover = cor_cover_antc & cor_cover_consq

            non_empty_intersection = False
            
            for candidate_clash in rules[idx:]:
                if candidate_clash.consequent[1] == default_class:
                    continue
                    
                cand_clas_cov_antc, _ = self.__dataframe.find_covered_by_rule_mask(candidate_clash)

                if any(cand_clas_cov_antc & cor_cover):
                    non_empty_intersection = True
                    break
                    
            new_rules.remove(rule) if non_empty_intersection == False else 0
            
        return new_rules