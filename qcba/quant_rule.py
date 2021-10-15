
import numpy as np
import copy
import pandas

from .interval_reader import IntervalReader


class LiteralCache:
    def __init__(self):
        self.__cache = {}

    def insert(self, literal, truth_values):
        self.__cache[literal] = truth_values

    def get(self, literal):
        return self.__cache[literal]

    def __contains__(self, literal):
        """function for using in
        on LiteralCache object
        """

        return literal in self.__cache.keys()


class QuantitativeDataFrame:

    def __init__(self, dataframe):
        if type(dataframe) != pandas.DataFrame:
            raise Exception("type of dataframe must be pandas.dataframe")

        self.__dataframe = dataframe
        self.__dataframe.iloc[:, -1] = self.__dataframe.iloc[:, -1].astype(str)

        # sorted and unique columns of the dataframe
        # saved as a numpy array
        self.__preprocessed_columns = self.__preprocess_columns(dataframe)

        # literal cache for computing rule statistics
        # - support and confidence
        self.__literal_cache = LiteralCache()

        # so that it doesn't have to be computed over and over
        self.size = dataframe.index.size

    @property
    def dataframe(self):
        return self.__dataframe

    def column(self, colname):
        return self.__preprocessed_columns[colname]

    def mask(self, vals):
        return self.__dataframe[vals]

    def find_covered_by_antecedent_mask(self, antecedent):
        """
        returns:
            mask - an array of boolean values indicating which instances
            are covered by antecedent
        """

        # todo: compute only once to make function faster
        dataset_size = self.__dataframe.index.size

        cummulated_mask = np.ones(dataset_size).astype(bool)

        for literal in antecedent:
            attribute, interval = literal

            # the column that concerns the
            # iterated attribute
            # instead of pandas.Series, grab the ndarray
            # using values attribute
            relevant_column = self.__dataframe[[
                attribute]].values.reshape(dataset_size)

            # this tells us which instances satisfy the literal
            current_mask = self.get_literal_coverage(literal, relevant_column)

            # add cummulated and current mask using logical AND
            cummulated_mask &= current_mask

        return cummulated_mask

    def find_covered_by_rule_mask(self, rule):
        """
        returns:
            covered_by_antecedent_mask:
                - array of boolean values indicating which
                dataset rows satisfy antecedent

            covered_by_consequent_mask:
                - array of boolean values indicating which
                dataset rows satisfy conseqeunt
        """

        dataset_size = self.__dataframe.index.size

        # initialize a mask filled with True values
        # it will get modified as futher literals get
        # tested

        # for optimization - create cummulated mask once
        # in constructor
        cummulated_mask = np.array([True] * dataset_size)

        for literal in rule.antecedent:
            attribute, interval = literal

            # the column that concerns the
            # iterated attribute
            # instead of pandas.Series, grab the ndarray
            # using values attribute
            relevant_column = self.__dataframe[[
                attribute]].values.reshape(dataset_size)

            # this tells us which instances satisfy the literal
            current_mask = self.get_literal_coverage(literal, relevant_column)

            # add cummulated and current mask using logical AND
            cummulated_mask &= current_mask

        instances_satisfying_antecedent_mask = cummulated_mask
        instances_satisfying_consequent_mask = self.__get_consequent_coverage_mask(
            rule)
        instances_satisfying_consequent_mask = instances_satisfying_consequent_mask.reshape(
            dataset_size)

        return instances_satisfying_antecedent_mask, instances_satisfying_consequent_mask

    def calculate_rule_statistics(self, rule):
        """calculates rule's confidence and
        support using efficient numpy functions


        returns:
        --------

            support:
                float

            confidence:
                float
        """

        dataset_size = self.__dataframe.index.size

        # initialize a mask filled with True values
        # it will get modified as futher literals get
        # tested

        # for optimization - create cummulated mask once
        # in constructor
        cummulated_mask = np.array([True] * dataset_size)

        for literal in rule.antecedent:
            attribute, interval = literal

            # the column that concerns the
            # iterated attribute
            # instead of pandas.Series, grab the ndarray
            # using values attribute
            relevant_column = self.__dataframe[[
                attribute]].values.reshape(dataset_size)

            # this tells us which instances satisfy the literal
            current_mask = self.get_literal_coverage(literal, relevant_column)

            # add cummulated and current mask using logical AND
            cummulated_mask &= current_mask

        instances_satisfying_antecedent = self.__dataframe[cummulated_mask].index
        instances_satisfying_antecedent_count = instances_satisfying_antecedent.size

        # using cummulated mask to filter out instances that satisfy consequent
        # but do not satisfy antecedent
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

        # instances satisfying consequent both antecedent and consequent

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

        if literal_key in self.__literal_cache:
            mask = self.__literal_cache.get(literal_key)
        else:
            mask = class_column == value

        return mask

    def get_literal_coverage(self, literal, values):
        """returns mask which describes the instances that
        satisfy the interval

        function uses cached results for efficiency
        """

        if type(values) != np.ndarray:
            raise Exception("Type of values must be numpy.ndarray")

        mask = []

        attribute, interval = literal

        literal_key = "{}={}".format(attribute, interval)

        # check if the result is already cached, otherwise
        # calculate and save the result
        if literal_key in self.__literal_cache:
            mask = self.__literal_cache.get(literal_key)
        else:
            mask = None

            if type(interval) == str:
                mask = np.array([val == interval for val in values])
            else:
                mask = interval.test_membership(values)

            self.__literal_cache.insert(literal_key, mask)

        # reshape mask into single dimension
        mask = mask.reshape(values.size)

        return mask

    def __preprocess_columns(self, dataframe):

        # covert to dict
        # column -> list
        # need to convert it to numpy array
        dataframe_dict = dataframe.to_dict(orient="list")

        dataframe_ndarray = {}

        for column, value_list in dataframe_dict.items():
            transformed_list = np.sort(np.unique(value_list))
            dataframe_ndarray[column] = transformed_list

        return dataframe_ndarray


class QuantitativeCAR:

    interval_reader = IntervalReader()

    def __init__(self, rule):
        self.antecedent = self.__create_intervals_from_antecedent(
            rule.antecedent)
        self.consequent = copy.copy(rule.consequent)

        self.confidence = rule.confidence
        self.support = rule.support
        self.rulelen = rule.rulelen
        self.rid = rule.rid

        # property which indicates wheter the rule was extended or not
        self.was_extended = False
        # literal which extended the rule
        self.extension_literal = None

        self.interval_reader = QuantitativeCAR.interval_reader

    def __create_intervals_from_antecedent(self, antecedent):
        interval_antecedent = []

        for literal in antecedent:
            attribute, value = literal

            # catch error if attribute is ordinal
            try:
                interval = QuantitativeCAR.interval_reader.read(value)

                interval_antecedent.append((attribute, interval))
            except:
                interval_antecedent.append((attribute, value))

        # return self.__sort_antecedent(interval_antecedent)
        return sorted(interval_antecedent)

    # def __sort_antecedent(self, antecedent):
    #     return sorted(antecedent)

    def update_properties(self, quant_dataframe):
        """updates rule properties using instance
        of QuantitativeDataFrame

        properties:
            support, confidence, rulelen

        """

        if quant_dataframe.__class__.__name__ != "QuantitativeDataFrame":
            raise Exception(
                "type of quant_dataframe must be QuantitativeDataFrame"
            )

        support, confidence = quant_dataframe.calculate_rule_statistics(self)

        self.support = support
        self.confidence = confidence
        # length of antecedent + length of consequent
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
        """
        precedence operator. Determines if this rule
        has higher precedence. Rules are sorted according
        to their confidence, support, length and id.
        """
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

    # def __lt__(self, other):
    #     """
    #     rule precedence operator
    #     """
    #     return not self > other

    def __eq__(self, other):
        return self.rid == other.rid
