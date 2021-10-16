import numpy as np
import copy
import pandas

from .range_iterator import RangeIterator
from .cache import Cache


class QuantitativeDataFrame:

    def __init__(self, dataframe):
        self.__dataframe = dataframe
        self.__dataframe.iloc[:, -1] = self.__dataframe.iloc[:, -1].astype(str)
        self.__preprocessed_columns = self.__preprocess(dataframe)
        self.__cache = Cache()
        self.size = dataframe.index.size

    @property
    def dataframe(self):
        return self.__dataframe

    def column(self, colname):
        return self.__preprocessed_columns[colname]

    def mask(self, vals):
        return self.__dataframe[vals]

    def __preprocess(self, df):
        processed = {}

        for key, value in df.to_dict(orient="list").items():
            processed[key] = np.sort(np.unique(value))

        return processed

    def find_covered_by_antecedent_mask(self, antecedent):
        cummulated_mask = np.ones(self.__dataframe.index.size).astype(bool)

        for literal in antecedent:
            attribute, interval = literal
            relevant_column = self.__dataframe[[
                attribute]].values.reshape(self.__dataframe.index.size)
            current_mask = self.get_literal_coverage(literal, relevant_column)
            cummulated_mask &= current_mask

        return cummulated_mask

    def find_covered_by_rule_mask(self, rule):
        cummulated_mask = np.array([True] * self.__dataframe.index.size)

        for literal in rule.antecedent:
            attribute, interval = literal
            relevant_column = self.__dataframe[[
                attribute]].values.reshape(self.__dataframe.index.size)
            current_mask = self.get_literal_coverage(literal, relevant_column)
            cummulated_mask &= current_mask

        instances_satisfying_antecedent_mask = cummulated_mask
        instances_satisfying_consequent_mask = self.__get_consequent_coverage_mask(
            rule)
        instances_satisfying_consequent_mask = instances_satisfying_consequent_mask.reshape(
            self.__dataframe.index.size)

        return instances_satisfying_antecedent_mask, instances_satisfying_consequent_mask

    def get_stats(self, rule):
        cummulated_mask = np.array([True] * self.__dataframe.index.size)

        for literal in rule.antecedent:
            attribute, interval = literal
            relevant_column = self.__dataframe[[
                attribute]].values.reshape(self.__dataframe.index.size)
            current_mask = self.get_literal_coverage(literal, relevant_column)
            cummulated_mask &= current_mask

        instances_satisfying_antecedent = self.__dataframe[cummulated_mask].index
        instances_satisfying_antecedent_count = instances_satisfying_antecedent.size
        instances_satisfying_consequent_mask = self.__get_consequent_coverage_mask(
            rule)
        instances_satisfying_consequent_mask = instances_satisfying_consequent_mask.reshape(
            self.__dataframe.index.size)

        instances_satisfying_consequent_and_antecedent = self.__dataframe[
            instances_satisfying_consequent_mask & cummulated_mask
        ].index

        instances_satisfying_consequent_and_antecedent_count = instances_satisfying_consequent_and_antecedent.size
        instances_satisfying_consequent_count = self.__dataframe[
            instances_satisfying_consequent_mask].index.size

        support = instances_satisfying_antecedent_count / self.__dataframe.index.size

        confidence = 0
        if instances_satisfying_antecedent_count != 0:
            confidence = instances_satisfying_consequent_and_antecedent_count / \
                instances_satisfying_antecedent_count

        return support, confidence

    def __get_consequent_coverage_mask(self, rule):
        attribute, value = rule.consequent
        mask = []
        class_column = self.__dataframe[[attribute]].values
        class_column = class_column.astype(str)

        if "{}={}".format(attribute, value) in self.__cache:
            mask = self.__cache.get("{}={}".format(attribute, value))
        else:
            mask = class_column == value

        return mask

    def get_literal_coverage(self, literal, values):
        mask = []

        attribute, interval = literal

        if "{}={}".format(attribute, interval) in self.__cache:
            mask = self.__cache.get("{}={}".format(attribute, interval))
        else:
            mask = None

            if type(interval) == str:
                mask = np.array([val == interval for val in values])
            else:
                mask = interval.test_membership(values)

            self.__cache.insert("{}={}".format(attribute, interval), mask)
        mask = mask.reshape(values.size)

        return mask


class QuantitativeCAR:

    range_iterator = RangeIterator()

    def __init__(self, rule):
        self.antecedent = self.__create_intervals_from_antecedent(
            rule.antecedent)
        self.consequent = copy.copy(rule.consequent)

        self.confidence = rule.confidence
        self.support = rule.support
        self.rulelen = rule.rulelen
        self.rid = rule.rid
        self.was_extended = False
        self.extension_literal = None

        self.range_iterator = QuantitativeCAR.range_iterator

    def __create_intervals_from_antecedent(self, antecedent):
        interval_antecedent = []

        for literal in antecedent:
            attribute, value = literal

            try:
                interval = QuantitativeCAR.range_iterator.read(value)

                interval_antecedent.append((attribute, interval))
            except:
                interval_antecedent.append((attribute, value))
        return sorted(interval_antecedent)

    def update_properties(self, df):

        support, confidence = df.get_stats(self)

        self.support = support
        self.confidence = confidence
        self.rulelen = len(self.antecedent) + 1

    def copy(self):
        return copy.deepcopy(self)

    def __deepcopy__(self, memo):

        copied = copy.copy(self)
        copied.antecedent = copy.deepcopy(self.antecedent)
        copied.consequent = copy.deepcopy(self.consequent)

        return copied

    def __repr__(self):
        arr = []
        for key, val in self.antecedent:
            if type(val) == str:
                arr.append("{}={}".format(key, val))
            else:
                arr.append("{}={}".format(key, val.string()))

        args = [
            "{" + ",".join(arr) + "}",
            "{" + self.consequent.string() + "}",
            self.support,
            self.confidence,
            self.rulelen,
            self.rid
        ]

        return "CAR {} => {} sup: {:.2f} conf: {:.2f} len: {}, id: {}".format(
            *args)

    def __gt__(self, other):
        if (self.confidence > other.confidence):
            return True
        elif (self.confidence == other.confidence and
              self.support > other.support):
            return True
        elif (self.confidence == other.confidence and
              self.support == other.support and
              self.rulelen < other.rulelen):
            return True
        elif(self.confidence == other.confidence and
             self.support == other.support and
             self.rulelen == other.rulelen and
             self.rid < other.rid):
            return True
        else:
            return False

    def __lt__(self, other):
        return not self > other

    def __eq__(self, other):
        return self.rid == other.rid
