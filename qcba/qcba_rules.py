
import numpy as np
import copy
import pandas

from .range_iterator import RangeIterator
from .cache import Cache


class QuantitativeDataFrame:

    def __init__(self, dataframe):
        self.__dataframe = dataframe
        self.__dataframe.iloc[:, -1] = self.__dataframe.iloc[:, -1].astype(str)
        self.__preprocessed_columns = self.__preprocess_columns(dataframe)
        self.__cache = Cache()
        self.size = dataframe.index.size

    @property
    def dataframe(self):
        return self.__dataframe

    def column(self, colname):
        return self.__preprocessed_columns[colname]

    def mask(self, vals):
        return self.__dataframe[vals]

    def find_covered_by_antecedent_mask(self, antecedent):
        dataset_size = self.__dataframe.index.size

        cummulated_mask = np.ones(dataset_size).astype(bool)

        for literal in antecedent:
            attribute, interval = literal
            relevant_column = self.__dataframe[[
                attribute]].values.reshape(dataset_size)
            current_mask = self.get_literal_coverage(literal, relevant_column)
            cummulated_mask &= current_mask

        return cummulated_mask

    def find_covered_by_rule_mask(self, rule):
        dataset_size = self.__dataframe.index.size
        cummulated_mask = np.array([True] * dataset_size)

        for literal in rule.antecedent:
            attribute, interval = literal
            relevant_column = self.__dataframe[[
                attribute]].values.reshape(dataset_size)
            current_mask = self.get_literal_coverage(literal, relevant_column)
            cummulated_mask &= current_mask

        instances_satisfying_antecedent_mask = cummulated_mask
        instances_satisfying_consequent_mask = self.__get_consequent_coverage_mask(
            rule)
        instances_satisfying_consequent_mask = instances_satisfying_consequent_mask.reshape(
            dataset_size)

        return instances_satisfying_antecedent_mask, instances_satisfying_consequent_mask

    def calculate_rule_statistics(self, rule):
        dataset_size = self.__dataframe.index.size
        cummulated_mask = np.array([True] * dataset_size)

        for literal in rule.antecedent:
            attribute, interval = literal
            relevant_column = self.__dataframe[[
                attribute]].values.reshape(dataset_size)
            current_mask = self.get_literal_coverage(literal, relevant_column)
            cummulated_mask &= current_mask

        instances_satisfying_antecedent = self.__dataframe[cummulated_mask].index
        instances_satisfying_antecedent_count = instances_satisfying_antecedent.size
        instances_satisfying_consequent_mask = self.__get_consequent_coverage_mask(
            rule)
        instances_satisfying_consequent_mask = instances_satisfying_consequent_mask.reshape(
            dataset_size)

        instances_satisfying_consequent_and_antecedent = self.__dataframe[
            instances_satisfying_consequent_mask & cummulated_mask
        ].index

        instances_satisfying_consequent_and_antecedent_count = instances_satisfying_consequent_and_antecedent.size
        instances_satisfying_consequent_count = self.__dataframe[
            instances_satisfying_consequent_mask].index.size

        support = instances_satisfying_antecedent_count / dataset_size

        confidence = 0
        if instances_satisfying_antecedent_count != 0:
            confidence = instances_satisfying_consequent_and_antecedent_count / \
                instances_satisfying_antecedent_count

        return support, confidence

    def __get_consequent_coverage_mask(self, rule):
        consequent = rule.consequent
        attribute, value = consequent

        class_column = self.__dataframe[[attribute]].values
        class_column = class_column.astype(str)

        literal_key = "{}={}".format(attribute, value)

        mask = []

        if literal_key in self.__cache:
            mask = self.__cache.get(literal_key)
        else:
            mask = class_column == value

        return mask

    def get_literal_coverage(self, literal, values):

        if type(values) != np.ndarray:
            raise Exception("Type of values must be numpy.ndarray")

        mask = []

        attribute, interval = literal

        literal_key = "{}={}".format(attribute, interval)
        if literal_key in self.__cache:
            mask = self.__cache.get(literal_key)
        else:
            mask = None

            if type(interval) == str:
                mask = np.array([val == interval for val in values])
            else:
                mask = interval.test_membership(values)

            self.__cache.insert(literal_key, mask)
        mask = mask.reshape(values.size)

        return mask

    def __preprocess_columns(self, dataframe):
        dataframe_dict = dataframe.to_dict(orient="list")

        dataframe_ndarray = {}

        for column, value_list in dataframe_dict.items():
            transformed_list = np.sort(np.unique(value_list))
            dataframe_ndarray[column] = transformed_list

        return dataframe_ndarray


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

    def update_properties(self, quant_dataframe):

        if quant_dataframe.__class__.__name__ != "QuantitativeDataFrame":
            raise Exception(
                "type of quant_dataframe must be QuantitativeDataFrame"
            )

        support, confidence = quant_dataframe.calculate_rule_statistics(self)

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
        ant = self.antecedent

        ant_string_arr = []
        for key, val in ant:
            if type(val) == str:
                ant_string_arr.append("{}={}".format(key, val))
            else:
                ant_string_arr.append("{}={}".format(key, val.string()))

        ant_string = "{" + ",".join(ant_string_arr) + "}"

        args = [
            ant_string,
            "{" + self.consequent.string() + "}",
            self.support,
            self.confidence,
            self.rulelen,
            self.rid
        ]

        text = "CAR {} => {} sup: {:.2f} conf: {:.2f} len: {}, id: {}".format(
            *args)

        return text

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
        """
        rule precedence operator
        """
        return not self > other

    def __eq__(self, other):
        return self.rid == other.rid
