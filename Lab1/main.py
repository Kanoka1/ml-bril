import pandas as pnd
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from Lab1.Bayes import Bayes
from Lab1.KNN import KNN
from data import Data
from utils import print_results


def test_bayes(data):
    bayes_sklearn = GaussianNB()
    bayes_sklearn.fit(data.train_x, data.train_y)
    bayes_sklearn_accuracy = accuracy_score(data.test_y, bayes_sklearn.predict(data.test_x))

    bayes_native = Bayes()
    bayes_native.fit(data.train_x, data.train_y)
    bayes_native_accuracy = accuracy_score(data.test_y, bayes_native.predict(data.test_x))
    return bayes_sklearn_accuracy, bayes_native_accuracy


def test_knn(data, k):
    my_kn_clf = KNN(k)
    my_kn_clf.fit(data.train_x, data.train_y)
    my_kn_clf_accuracy = accuracy_score(data.test_y, my_kn_clf.predict(data.test_x))

    sklearn_kn_clf = KNeighborsClassifier(k)
    sklearn_kn_clf.fit(data.train_x, data.train_y)
    sklearn_kn_clf_accuracy = accuracy_score(data.test_y, sklearn_kn_clf.predict(data.test_x))

    return sklearn_kn_clf_accuracy, my_kn_clf_accuracy


def main():
    raw_data = pnd.read_table('../data/segmentation.test', sep=',', header=None, lineterminator='\n')
    data = Data(raw_data, 0.7)

    bayes_sklearn_accuracy, bayes_native_accuracy = test_bayes(data)
    sklearn_kn_clf_accuracy, my_kn_clf_accuracy = test_knn(data, 10)

    text = 'bayes,{:0.4f}%,{:0.4f}%\n' \
           'knn,{:0.4f}%,{:0.4f}%\n'.format(bayes_sklearn_accuracy,
                                            bayes_native_accuracy,
                                            sklearn_kn_clf_accuracy,
                                            my_kn_clf_accuracy)

    print_results(text)


main()
