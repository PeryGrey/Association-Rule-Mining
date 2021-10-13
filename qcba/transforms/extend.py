import pandas
import numpy as np
import math

from ..data_structures import QuantitativeDataFrame, Interval

class Extend:    
    def __init__(self, dataframe):
        self.__dataframe = dataframe
        
    def transform(self, rules):        
        r = [ rule.copy() for rule in rules ]
        return [self.__extend_rule(rule) for rule in r]

    def __extend_rule(self, rule):
        current_best = rule        
        current_best.update_properties(self.__dataframe)
        
        while True:
            extension_succesful = False

            for c in self.__get_extensions(current_best):
                c.update_properties(self.__dataframe)                
                d_conf = c.confidence - current_best.confidence
                ds = c.support - current_best.support
                
                if d_conf >= 0 and ds > 0:                
                    current_best = c
                    extension_succesful = True
                    break
                
                if d_conf >= -0.01:                    
                    while True:
                        e = self.get_beam_extensions(c)
                        if not e:
                            break
                            
                        c.update_properties(self.__dataframe)
                        e.update_properties(self.__dataframe)

                        d_conf = e.confidence - current_best.confidence
                        ds = e.support - current_best.support

                        if d_conf >= 0 and ds > 0:      
                            current_best = e
                            extension_succesful = True
                        elif d_conf >= -0.01:
                            continue              
                        else:
                            break

                    if extension_succesful:
                        break  
                else:
                    continue
            if not extension_succesful:
                break
        return current_best

    def __get_extensions(self, rule):
        def inner_loop(el, lit):
            cr = rule.copy()
            idx = cr.antecedent.index(lit)
            cr.antecedent[idx] = el
            cr.was_extended = True 
            cr.extended_literal = el
            return cr

        extended_rules = []
        
        for literal in rule.antecedent:
            attribute, interval = literal            
            neighborhood = self.__get_direct_extensions(literal)
            extended_rules = [inner_loop(el, literal) for el in neighborhood]

        extended_rules.sort(reverse=True)
             
        return extended_rules
            
    
    def __get_direct_extensions(self, literal):        
        attribute, interval = literal

        if type(interval) == str:
            return [literal]
        
        vals = self.__dataframe.column(attribute)

        mask = interval.test_membership(vals)
        mi = np.where(mask)[0]

        first_index_modified = mi[0] - 1
        last_index_modified = mi[-1] + 1

        extensions = []

        if not (first_index_modified < 0):
            new_left_bound = vals[first_index_modified]
            temp_interval = Interval(new_left_bound, interval.maxval, True, interval.right_inclusive)
            extensions.append((attribute, temp_interval))

        if not (last_index_modified > len(vals) - 1):
            new_right_bound = vals[last_index_modified]
            temp_interval = Interval(interval.minval, new_right_bound, interval.left_inclusive, True)
            extensions.append((attribute, temp_interval))

        return extensions

    def get_beam_extensions(self, rule):
        if not rule.was_extended:
            return None

        literal = rule.extended_literal        
        el = self.__get_direct_extensions(literal)
        
        if not el:
            return None
        
        cr = rule.copy()        
        li = cr.antecedent.index(literal)
        cr.antecedent[li], cr.extended_literal = el[0], el[0]
        cr.was_extended = True
        return cr
        