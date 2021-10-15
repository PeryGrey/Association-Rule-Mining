from .transforms import (
    Refit,
    PruneLiterals,
    Trim,
    Extend,
    PostPrune,
    Prune_Overlap
)


class QCBATransformation:
    def __init__(self, datset, transaction_based_drop=True):
        self.transaction_based_drop = transaction_based_drop

        self.refitter = Refit(datset)
        self.literal_pruner = PruneLiterals(datset)
        self.trimmer = Trim(datset)
        self.extender = Extend(datset)
        self.post_pruner = PostPrune(datset)
        self.overlap_pruner = Prune_Overlap(datset)

    def __refit(self, rules):
        print('1) Doing refitting')
        return self.refitter.transform(rules)

    def __prune_literals(self, rules):
        print("2) Doing literal pruning")
        return self.literal_pruner.transform(rules)

    def __trim(self, rules):
        print("3) Doing trimming")
        return self.trimmer.transform(rules)

    def __extend(self, rules):
        print("4) Doing extension")
        return self.extender.transform(rules)

    def transform(self, rules, stages={}):
        rules = rules
        rules = self.__refit(rules) if stages['refitting'] else rules
        rules = self.__prune_literals(rules) if stages['literal_pruning'] else rules
        rules = self.__trim(rules) if stages['trimming'] else rules
        rules = self.__extend(rules) if stages['extension'] else rules

        print("5) Doing post pruning")
        rules, default_class = self.post_pruner.transform(rules)
        print("6) Doing overlap pruning")
        rules = self.overlap_pruner.transform(rules, default_class, transaction_based=stages["based_drop"])
        return rules, default_class
