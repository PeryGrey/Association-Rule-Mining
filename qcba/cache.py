class Cache:
    def __init__(self):
        self.__cache = {}

    def insert(self, literal, truth_values):
        self.__cache[literal] = truth_values

    def get(self, literal):
        return self.__cache[literal]

    def __contains__(self, literal):
        return literal in self.__cache.keys()
