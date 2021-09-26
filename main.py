from data_structures import TransactionDB
from cba import CBA
import pandas as pd
import random
import numpy as np

data_train = pd.read_csv("discretized-heart.csv")
data_test = pd.read_csv("discretized-heart.csv")

block_size = int(len(data_train) / 10)
split_point = [k * block_size for k in range(0, 10)]
split_point.append(len(data_train))
accuracies = []

for k in range(len(split_point)-1):
    print("\nRound %d:" % k)
    test_dataset = data_test[split_point[k]:split_point[k+1]]
    train_dataset = data_train[0:split_point[k]].append(
        data_train[split_point[k+1]:])
    txns_train = TransactionDB.from_DataFrame(train_dataset)
    txns_test = TransactionDB.from_DataFrame(test_dataset)
    cba = CBA(support=0.01, confidence=0.5, algorithm="m1")
    cba.fit(txns_train)
    accuracy = cba.rule_model_accuracy(txns_test)
    accuracies.append(accuracy)
print(accuracies)
print("Mean accuracy = ", np.mean(accuracies))
