import pandas as pnd
from data import Data
from Lab2.decision_tree import DecisionTree

import utils


def print_results(experiment_number, train_size, dtc_accuracy, rfc_accuracy):
    text = '{}, {}, {}, {}\n'.format(experiment_number, train_size * 100, dtc_accuracy * 100, rfc_accuracy * 100)
    utils.print_results(text)


def main():
    raw_data = pnd.read_table('../data/segmentation.test', sep=',', header=None, lineterminator='\n')

    open('results/results.txt', 'w').close()

    set_test_coefficients = [0.6, 0.7, 0.8, 0.9]
    experiments = []

    for c in set_test_coefficients:
        experiments.append(DecisionTree(Data(raw_data, c)))

    pairs = zip(set_test_coefficients, experiments)

    for c, exp in pairs:
        d_accuracy, r_accuracy = exp.run()
        print_results(1, c, d_accuracy, r_accuracy)


main()
