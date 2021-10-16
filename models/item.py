class Item():
    """ Item class for representing attribute-value pair
    and one item in transaction or antecedent.
    """

    def __init__(self, attribute, value):
        # convert attribute and value so that
        # Item("a", 1) == Item("a", "1")
        self.attribute = repr(attribute) if type(
            attribute) != str else attribute
        self.value = repr(value) if type(value) != str else value

    def __get_tuple(self):
   
        return (self.attribute, self.value)

    def __getitem__(self, idx):
        """Method for accessing Item as a tuple"""
        item = self.__get_tuple()
        return item[idx]

    def __hash__(self):
        """Two Items with the same attribute and value
        have identical hash value.
        """
        return hash(self.__get_tuple())

    def __eq__(self, other):
    
        return hash(self) == hash(other)

    def __repr__(self):
  
        return "Item{{{}}}".format(self.__get_tuple())

    def string(self):
        return "{}={}".format(*self)
