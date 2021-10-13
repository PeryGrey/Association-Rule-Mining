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
        return self.prune([ rule.copy() for rule in rules ])

    def get_most_class(self):        
        index_counts, possible_classes = pd.factorize(self.__dataframe.dataframe.iloc[:, -1].values)
        counts = np.bincount(index_counts)
        counts_max = counts.max()
        most_frequent_classes = possible_classes[counts == counts_max]
        
        return most_frequent_classes[0], counts_max
    
    def get_most_class_np(self, ndarray):
        unique, pos = np.unique(ndarray, return_inverse=True) 
        counts = np.bincount(pos)                  
        maxpos = counts.argmax()                      

        return (unique[maxpos], counts[maxpos])
        
    def prune(self, rules):
        dataset = self.__dataframe.dataframe.index.values 
        dataset_mask = [ True ] * dataset.size
        
        cutoff_rule = rules[-1]
        cutoff_class, cutoff_class_count = self.get_most_class()
        
        default_class = cutoff_class
        total_errors_without_default = 0        
        lowest_total_error = dataset.size - cutoff_class_count
        
        rules.sort(reverse=True)
        
        for rule in rules:
            cover_ant, cover_consq = self.__dataframe.find_covered_by_rule_mask(rule)
            correctly_covered = cover_ant & cover_consq
            if not any(correctly_covered):
                rules.remove(rule)
            else:
                misclassified = np.sum(cover_ant & dataset_mask) - np.sum(correctly_covered & dataset_mask)
                
                total_errors_without_default += misclassified
                dataset_mask = dataset_mask & np.logical_not(cover_ant)

                modified_dataset = dataset[dataset_mask]
                class_values = self.__dataframe.dataframe.iloc[:,-1][dataset_mask].values

                default_class, default_class_count = self.__dataframe.dataframe.iloc[1,-1], 0
                
                if len(class_values) > 0:
                    default_class, default_class_count = self.get_most_class_np(class_values)

                default_rule_error = np.sum(dataset_mask) - default_class_count
                total_errors_with_default = default_rule_error + total_errors_without_default
                
                if total_errors_with_default < lowest_total_error:
                    cutoff_rule = rule
                    lowest_total_error = total_errors_with_default
                    cutoff_class = default_class
        
        rules_pruned = rules[:rules.index(cutoff_rule)+1]
        return rules_pruned, cutoff_class
        
        
        