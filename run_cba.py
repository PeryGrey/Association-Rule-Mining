from models import TransactionDB
from cba import ClassBasedAssoc
import pandas as pd
import random
import numpy as np
from sklearn.utils import shuffle
import evaluate as Evaluate
import time

# path = "tic-tac-toe-endgame"
path = "iris"

data_train = pd.read_csv(f"datasets/cba-datasets/discretized-{path}.csv")
data_test = pd.read_csv(f"datasets/cba-datasets/discretized-{path}.csv")
data_train = shuffle(data_train)
data_test = shuffle(data_test)

block_size = int(len(data_train) / 10)
split_point = [k * block_size for k in range(0, 10)]
split_point.append(len(data_train))
accuracies = []

for k in range(len(split_point) - 1):
    print("\nRound %d:" % k)
    test_dataset = data_test[split_point[k] : split_point[k + 1]]
    train_dataset = data_train[0 : split_point[k]].append(
        data_train[split_point[k + 1] :]
    )
    txns_train = TransactionDB.from_DataFrame(train_dataset)
    txns_test = TransactionDB.from_DataFrame(test_dataset)
    time1 = time.time()
    cba = ClassBasedAssoc(support=0.01, confidence=0.5, classifier="m1")
    cars = cba.generateCARS(txns_train)
    classifier = cba.buildClassifier(cars,txns_train)
    time2 = time.time()
    print("time: ", time2 - time1)
    accuracy = Evaluate.evaluate(classifier, txns_test)
    accuracies.append(accuracy)
print("")
print(accuracies)
print("Mean accuracy = ", np.mean(accuracies))
