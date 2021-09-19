from data_structures import TransactionDB
from m1algorithm import M1Algorithm
from m2algorithm import M2Algorithm
from rule_generation import createCARs, top_rules
import pandas as pd


data_train = pd.read_csv("titanic.csv")
data_test = pd.read_csv("titanic.csv")

txns_train = TransactionDB.from_DataFrame(data_train)
txns_test = TransactionDB.from_DataFrame(data_test)

# get the best association rules
rules = top_rules(txns_train.string_representation)

# convert them to class association rules
cars = createCARs(rules)
print(cars)

classifier = M1Algorithm(cars, txns_train).build()
# classifier = M2Algorithm(cars, txns_train).build()

accuracy = classifier.test_transactions(txns_test)
print(accuracy)
