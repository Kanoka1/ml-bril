import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from matplotlib.colors import Normalize


class MidpointNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


class Data:
    def __init__(self, data, train_size):
        self.x = data.values[:, 4:]
        self.y = data.values[:, 0]
        self.train_size = train_size
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(self.x, self.y,
                                                                                train_size=train_size)
        self.y_names = set(self.y)


def test_kernel_functions(data):
    C = 1.0
    svc = svm.SVC(kernel='linear', C=C).fit(data.train_x, data.train_y)
    lin_svc = svm.LinearSVC(C=C).fit(data.train_x, data.train_y)
    rbf_svc = svm.SVC(kernel='rbf', C=C).fit(data.train_x, data.train_y)
    sigmoid_svc = svm.SVC(kernel='sigmoid', C=C).fit(data.train_x, data.train_y)
    poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(data.train_x, data.train_y)

    titles = ['SVC with linear kernel',
              'LinearSVC (linear kernel)',
              'SVC with rbf kernel',
              'SVC with sigmoid kernel',
              'SVC with poly kernel']

    for i, clf in enumerate((svc, lin_svc, rbf_svc, sigmoid_svc, poly_svc)):
        pred = clf.predict(data.test_x)
        print('Accuracy for {}: {:.2%}'.format(titles[i], metrics.accuracy_score(data.test_y, pred)))


def linear_c_test(data):
    C_range = np.logspace(-4, 5, 10)
    params = dict(C=C_range)

    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    grid = GridSearchCV(svm.SVC(kernel='linear'), param_grid=params, cv=cv)
    grid.fit(data.x, data.y)

    x = grid.cv_results_['mean_test_score']
    print(x)
    # the histogram of the data
    plt.hist(x, facecolor='g', alpha=0.75)

    plt.xlabel('C')
    plt.ylabel('accuracy')
    plt.title('SVC Linera C changing')
    plt.grid(True)
    plt.xticks(np.arange(len(C_range)), C_range, rotation=45)
    plt.show()

def test_gamma(data):
    svc_g01 = svm.SVC(kernel='poly', gamma=0.1).fit(data.train_x, data.train_y)
    svc_g1 = svm.SVC(kernel='poly', gamma=1).fit(data.train_x, data.train_y)
    svc_g10 = svm.SVC(kernel='poly', gamma=10).fit(data.train_x, data.train_y)
    svc_g100 = svm.SVC(kernel='poly', gamma=100).fit(data.train_x, data.train_y)

    for i, clf in enumerate((svc_g01, svc_g01, svc_g10, svc_g100)):
        pred = clf.predict(data.test_x)
        print('Accuracy for {}: {:.2%}'.format(titles[i], metrics.accuracy_score(data.test_y, pred)))


def poly_c_gamma_test(data):
    C_range = np.logspace(-2, 7, 10)
    gamma_range = np.logspace(-9, 0, 10)
    param_grid = dict(gamma=gamma_range, C=C_range)
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    grid = GridSearchCV(svm.SVC(kernel='poly'), param_grid=param_grid, cv=cv)
    grid.fit(data.x, data.y)

    scores = grid.cv_results_['mean_test_score'].reshape(len(C_range),
                                                         len(gamma_range))

    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,
               norm=MidpointNormalize(vmin=0.2, midpoint=0.86))
    plt.xlabel('gamma')
    plt.ylabel('C')
    plt.colorbar()
    plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
    plt.yticks(np.arange(len(C_range)), C_range)
    plt.title('Validation accuracy')
    plt.show()


def poly_c_coef_test(data):
    C_range = np.logspace(-3, 4, 8)
    coef0_range = np.logspace(-4, 3, 8)
    param_grid = dict(coef0=coef0_range, C=C_range)
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    grid = GridSearchCV(svm.SVC(kernel='poly'), param_grid=param_grid, cv=cv)
    grid.fit(data.x, data.y)

    scores = grid.cv_results_['mean_test_score'].reshape(len(C_range),
                                                         len(coef0_range))

    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,
               norm=MidpointNormalize(vmin=0.2, midpoint=0.86))
    plt.xlabel('coef0')
    plt.ylabel('C')
    plt.colorbar()
    plt.xticks(np.arange(len(coef0_range)), coef0_range, rotation=45)
    plt.yticks(np.arange(len(C_range)), C_range)
    plt.title('Validation accuracy')
    plt.show()


def poly_c_degre_test(data):
    C_range = np.logspace(-2, 5, 8)
    degree_range = np.linspace(1, 4.5, 8)
    param_grid = dict(degree=degree_range, C=C_range)
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    grid = GridSearchCV(svm.SVC(kernel='poly'), param_grid=param_grid, cv=cv)
    grid.fit(data.x, data.y)

    scores = grid.cv_results_['mean_test_score'].reshape(len(C_range), len(degree_range))

    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,
               norm=MidpointNormalize(vmin=0.2, midpoint=0.86))
    plt.xlabel('degree')
    plt.ylabel('C')
    plt.colorbar()
    plt.xticks(np.arange(len(degree_range)), degree_range, rotation=45)
    plt.yticks(np.arange(len(C_range)), C_range)
    plt.title('Validation accuracy')
    plt.show()


def get_svc_accuracy(fst_length, scnd_length):
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    grid = GridSearchCV(svm.SVC(kernel='poly'), param_grid=param_grid, cv=cv)
    grid.fit(data.x, data.y)

    return grid.cv_results_['mean_test_score'].reshape(scnd_length, fst_length)


def draw(scores, fst_range, scnd_range, fst_name, scnd_name):
    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,
               norm=MidpointNormalize(vmin=0.2, midpoint=0.86))
    plt.xlabel(fst_name)
    plt.ylabel(scnd_name)
    plt.colorbar()
    plt.xticks(np.arange(len(fst_range)), fst_range, rotation=45)
    plt.yticks(np.arange(len(scnd_range)), scnd_range)
    plt.title('Validation accuracy')
    plt.show()


def load_data():
    raw_data = pd.read_table('../../data/segmentation.data', sep=',', header=None, lineterminator='\n')
    data = Data(raw_data, 0.7)
    return data