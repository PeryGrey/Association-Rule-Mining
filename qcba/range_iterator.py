import re
import numpy as np

def form_ranges(minimumvalue, maximumvalue, includeleft, includeright):
    def temp_func(value_to_compare):
        if isValueGreater(value_to_compare, minimumvalue, includeleft) and isValueLess(value_to_compare, maximumvalue, includeright):
            return True
        else:
            return False

    return temp_func


def isValueGreater(d, e, boundary):
    if boundary:
        if d >= e:
            return True
    elif d > e:
        return True

    return False


def isValueLess(d, e, boundary):
    if boundary:
        if d <= e:
            return True
    elif d < e:
        return True

    return False


class Range:

    def __init__(self, minimumvalue, maximumvalue, left_boundary, right_boundary):
        self.minimumvalue = minimumvalue
        self.maximumvalue = maximumvalue
        self.left_boundary = left_boundary
        self.right_boundary = right_boundary

        self.left_bracket = "<" if left_boundary else "("
        self.right_bracket = ">" if right_boundary else ")"

        self.__membership_func = np.vectorize(
            form_ranges(self.minimumvalue, self.maximumvalue,
                        self.left_boundary, self.right_boundary)
        )

    def __hash__(self):
        return hash(repr(self))

    def __eq__(self, other):
        return hash(self) == hash(other)

    def refit(self, vals):

        processed_list = np.array(vals)[self.test_membership(np.array(vals))]
        if(len(processed_list) < 1):
            minimum, maximum = 0, 0
        else:
            minimum, maximum = min(processed_list), max(processed_list)

        return Range(minimum, maximum, True, True)

    def test_membership(self, value):
        return self.__membership_func(value)

    def isin(self, value):
        return self.test_membership([value])[0]

    def overlaps_with(self, other):
        return self.isin(other.minimumvalue) or self.isin(other.maximumvalue) or other.isin(self.minimumvalue) or other.isin(self.maximumvalue)

    def string(self):
        return "{}{};{}{}".format(self.left_bracket, self.minimumvalue, self.maximumvalue, self.right_bracket)

    def __repr__(self):
        return "Range[{}{};{}{}]".format(self.left_bracket, self.minimumvalue, self.maximumvalue, self.right_bracket)


class RangeIterator():

    def __init__(self):
        self.__open_bracket = "(", ")"
        self.__closed_bracket = "<", ">"
        self.__infinity_symbol = "-inf", "+inf"
        self.__decimal_separator = "."
        self.__members_separator = ";"
        self.initialize_reader()

    def initialize_reader(self):

        regex_range_string = "{}{}{}{}{}".format(
            "({}|{})".format(
                re.escape(self.open_bracket[0]),
                re.escape(self.closed_bracket[0])
            ),
            "(\-?\d+(?:{}(?:\d)+)?|{})".format(
                re.escape(self.decimal_separator),
                re.escape(self.infinity_symbol[0]),
            ),
            "{}".format(
                re.escape(self.members_separator)
            ),
            "(\-?\d+(?:{}(?:\d)+)?|{})".format(
                re.escape(self.decimal_separator),
                re.escape(self.infinity_symbol[1]),
            ),
            "({}|{})".format(
                re.escape(self.open_bracket[1]),
                re.escape(self.closed_bracket[1])
            )
        )

        self.__interval_regex = re.compile(regex_range_string)

    def read(self, interval_string):
        bracket_left, minimumvalue, maximumvalue, bracket_right = self.__interval_regex.findall(
            interval_string)[0]

        return Range(
            float(
                minimumvalue) if minimumvalue != self.infinity_symbol[0] else np.NINF,
            float(
                maximumvalue) if maximumvalue != self.infinity_symbol[1] else np.PINF,
            True if bracket_left == self.closed_bracket[0] else False,
            True if bracket_right == self.closed_bracket[1] else False
        )

    @property
    def open_bracket(self):
        return self.__open_bracket

    @open_bracket.setter
    def open_bracket(self, val):
        self.__open_bracket = val
        return self

    @property
    def closed_bracket(self):
        return self.__closed_bracket

    @closed_bracket.setter
    def closed_bracket(self, val):
        self.__closed_bracket = val
        return self

    @property
    def infinity_symbol(self):
        return self.__infinity_symbol

    @infinity_symbol.setter
    def infinity_symbol(self, val):
        self.__infinity_symbol = val
        return self

    @property
    def decimal_separator(self):
        return self.__decimal_separator

    @decimal_separator.setter
    def decimal_separator(self, val):
        self.__decimal_separator = val
        return self

    @property
    def members_separator(self):
        return self.__members_separator

    @members_separator.setter
    def members_separator(self, val):
        self.__members_separator = val
        return self
