import time
import fim
from models import Consequent, Item, Antecedent, ClassAssocationRule
import logging


def convertToCARs(rules):  # coverts apriori rules to CARS format
    CARs = []

    for rule in rules:
        rhs_tmp, lhs_tmp, support, confidence = rule
        rhs = Consequent(*rhs_tmp.split(":=:"))

        # so that the order of items in antecedent is always the same
        lhs_tmp = sorted(list(lhs_tmp))
        lhs_items = [Item(*i.split(":=:")) for i in lhs_tmp]
        lhs = Antecedent(lhs_items)  # convert the antecedents to dict form

        CAR = ClassAssocationRule(
            lhs, rhs, support=support, confidence=confidence
        )  # convert to CAR
        CARs.append(CAR)

 
    return CARs


def generateARs(
    transactionDB, support=1, confidence=50, maxlen=10, **kwargs
):  # generates Assoc. rules using apriori and converts them to CARS

    appear = transactionDB.appeardict
    print("Mining top association rules")
    rules = fim.apriori(
        transactionDB.string_representation,
        supp=support,
        conf=confidence,
        mode="o",
        target="r",
        report="sc",
        appear=appear,
        **kwargs,
        zmax=maxlen,
    )

    print("No of association rules from apriori: ", len(rules))
    # print("Rule example: ", rules[0])

    return rules
