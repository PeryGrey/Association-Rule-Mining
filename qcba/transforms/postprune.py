import collections
from scipy import stats

import pandas
import pandas as pd
import numpy as np

from ..data_structures import QuantitativeDataFrame, Interval


class PostPrune:
    
    def __init__(self, quantitative_dataset):
        self.__dataframe = quantitative_dataset
        
    def transform(self, rules):
        copied_rules = [rule.copy() for rule in rules ]
        return self.prune(copied_rules)
        
    def preprocess_dataframe(self):
        return self.__dataframe.dataframe.index.values

    def get_most_frequent_class(self):        
        index_counts, possible_classes = pd.factorize(self.__dataframe.dataframe.iloc[:, -1].values)
        counts = np.bincount(index_counts)
        most_frequent_classes = possible_classes[counts == counts.max()]
        return most_frequent_classes[0], counts.max()
    
    def get_most_frequent_from_numpy(self, ndarray):
        unique, pos = np.unique(ndarray, return_inverse=True) 
        counts = np.bincount(pos)                  
        maxpos = counts.argmax()
        return (unique[maxpos], counts[maxpos])
        
    def prune(self, rules):        
        dataset = self.preprocess_dataframe()
        dataset_len = dataset.size
        dataset_mask = [ True ] * dataset_len
        
        cutoff_rule = rules[-1]
        cutoff_class, cutoff_class_count = self.get_most_frequent_class()
        
        default_class = cutoff_class
        total_errors_without_default = 0        
        lowest_total_error = dataset_len - cutoff_class_count

        rules.sort(reverse=True)
        
        for rule in rules:
            covered_antecedent, covered_consequent = self.__dataframe.find_covered_by_rule_mask(rule)
            correctly_covered = covered_antecedent & covered_consequent
            
            if not any(correctly_covered):
                rules.remove(rule)
            else:
                misclassified = np.sum(covered_antecedent & dataset_mask) - np.sum(correctly_covered & dataset_mask)
                
                total_errors_without_default += misclassified
                dataset_mask = dataset_mask & np.logical_not(covered_antecedent)

                modified_dataset = dataset[dataset_mask]
                class_values = self.__dataframe.dataframe.iloc[:,-1][dataset_mask].values

                default_class, default_class_count = self.__dataframe.dataframe.iloc[1,-1], 0
                
                if len(class_values) > 0:
                    default_class, default_class_count = self.get_most_frequent_from_numpy(class_values)
                
                default_rule_error = np.sum(dataset_mask) - default_class_count
                total_errors_with_default = default_rule_error + total_errors_without_default

                if total_errors_with_default < lowest_total_error:
                    cutoff_rule = rule
                    lowest_total_error = total_errors_with_default
                    cutoff_class = default_class
        
        index_to_cut = rules.index(cutoff_rule)
        rules_pruned = rules[:index_to_cut+1]
        
        return rules_pruned, cutoff_class
        
        
        