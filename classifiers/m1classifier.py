import collections
from models.classifier import Classifier
import time
import random
from collections import Counter
from models import ClassAssocationRule, Antecedent, Consequent


class M1Classifier:
    """M1 Algorithm implementation."""

    def __init__(self, rules, dataset):
        self.rules = rules
        self.dataset = dataset
        self.y = dataset.class_labels

    def train(self):
        self.stepOne()
        classifier_rules, default_classes, total_errors, class_distribution, classdist_keys = self.stepTwo()
        clf = self.stepThree(classifier_rules, default_classes,
                             total_errors, class_distribution, classdist_keys)
        return clf

    def stepOne(self):
        self.rules.sort(reverse=True)

    def stepTwo(self):
        classifier_rules = []

        default_classes = []

        default_classes_errors = []

        # storing rule errors
        rule_errors = []

        total_errors = []

        class_distribution = collections.Counter(self.y)
        classdist_keys = list(class_distribution.keys())

        dataset = set(self.dataset)
        dataset_len = len(dataset)

        dataset_len_updated = dataset_len

        for rule in self.rules:
            if dataset_len_updated <= 0:
                break

            # storing data points that have been covered by current rule
            temp = set()

            temp_len = 0
            # number of rule that satisfy both lhs and rhs ofcurrent rule
            temp_satisfies_rhs_counter = 0

            for data in dataset:

                if rule.antecedent <= data:
                    temp.add(data)
                    temp_len += 1

                    # Checkingif data satisfies its consequent
                    if rule.consequent == data.class_val:
                        temp_satisfies_rhs_counter += 1
                        rule.marked = True

            # if rule satisfied at least one consequent
            if rule.marked:
                classifier_rules.append(rule)
                # remove covered rule from dataset
                dataset -= temp
                dataset_len_updated -= temp_len

                class_distribution = collections.Counter(
                    map(lambda d: d.class_val.value, dataset)
                )

                # finding the default class for uncovered cases
                most_common_tuple = class_distribution.most_common(1)
                most_common_counter = 0
                most_common_label = "None"

                try:
                    most_common_tuple = most_common_tuple[0]
                    most_common_counter = most_common_tuple[1]
                    most_common_label = most_common_tuple[0]
                except IndexError:
                    pass

                default_classes.append(most_common_label)

                # calculating number of errors the rule will make
                rule_errors.append(temp_len - temp_satisfies_rhs_counter)
                # calculating error for default class
                default_class_err = dataset_len_updated - most_common_counter

                err_counter = default_class_err

                default_classes_errors.append(err_counter)

                total_errors.append(err_counter + sum(rule_errors))

        return classifier_rules, default_classes, total_errors, class_distribution, classdist_keys

    def stepThree(self, classifier_rules, default_classes, total_errors, class_distribution, classdist_keys):
        print("Total no of rules in M1 before discarding: ", len(classifier_rules))
        if len(total_errors) != 0:
            min_errors = min(total_errors)

            # finding the index of rule with smallest number of errors (Threshold for discarding the remaining rules)
            threshold_idx = total_errors.index(min_errors)
            # discarding all rules after threshold
            final_classifier_rules = classifier_rules[: threshold_idx + 1]
            print("No of rules in M1 after discarding: ",
                  len(final_classifier_rules))
            default_class = default_classes[threshold_idx]
            print("Default Class: ", default_class)

            # creating the final classifier
            clf = Classifier()
            clf.rules = final_classifier_rules
            clf.default_class = default_class
            clf.default_class_attribute = classdist_keys[0][0]

        return clf
