import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr
from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D
from data import Data

def visualize_data(X, y):

    plt.figure(figsize=(8, 6))

    for label, color in zip(('BRICKFACE', 'SKY', 'FOLIAGE', 'CEMENT', 'WINDOW', 'PATH', 'GRASS'),
                            ('crimson', 'lightblue', 'darkgreen', 'gray', 'orange', 'black', 'lightgreen')):
        R = pearsonr(X[:, 11][y == label], X[:, 13][y == label])
        plt.scatter(x=X[:, 11][y == label],
                    y=X[:, 13][y == label],
                    color=color,
                    alpha=0.7,
                    label='{:}, R={:.2f}'.format(label, R[0])
                    )

    plt.title('Image Segmentation data visualization')
    plt.xlabel('value mean')
    plt.ylabel('hue mean')
    plt.legend(loc='best')

    plt.show()


def visualize_3d_data(X, y):

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    colors = ['crimson', 'lightblue', 'darkgreen', 'gray', 'orange', 'black', 'lightgreen']

    for label, color in zip(data.y_names, colors):
        ax.scatter(X[:, 12][y == label],
                   X[:, 11][y == label],
                   X[:, 13][y == label],
                   color=color,
                   s=40,
                   alpha=0.7,
                   label=label)

    ax.set_xlabel('saturation mean')
    ax.set_ylabel('value mean')
    ax.set_zlabel('hue mean')
    plt.legend(loc='best')
    plt.title('Image Segmentation data visualization')

    plt.show()


def visualize_lda(data):
    lda_clf = LDA()
    lda_transform = lda_clf.fit(data.train_x, data.train_y).transform(data.train_x)

    colors = ['crimson', 'lightblue', 'darkgreen', 'gray', 'orange', 'black', 'lightgreen']

    plt.figure()
    for color, target_name in zip(colors, data.y_names):
        plt.scatter(lda_transform[data.train_y == target_name, 2], lda_transform[data.train_y == target_name, 1],
                    alpha=.8, color=color,
                    label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('LDA of Image Segmentation dataset')

    plt.show()


def test_lda(data):
    lda_clf = LDA()
    lda_clf.fit(data.train_x, data.train_y)

    lda_predict = lda_clf.predict(data.train_x)
    print('LDA')
    print('Classification accuracy for train data = {:.2%}'.format(metrics.accuracy_score(data.train_y, lda_predict)))

    test_result = lda_clf.predict(data.test_x)

    print('Classification accuracy for test data =  {:.2%}'.format(metrics.accuracy_score(data.test_y, test_result)))


def test_qda(data):
    qda_clf = QDA()
    qda_clf.fit(data.train_x, data.train_y)

    qda_predict = qda_clf.predict(data.train_x)
    print('QDA')
    print('Classification accuracy for train data = {:.2%}'.format(metrics.accuracy_score(data.train_y, qda_predict)))

    test_result = qda_clf.predict(data.test_x)

    print('Classification accuracy for test data = {:.2%}'.format(metrics.accuracy_score(data.test_y, test_result)))


def main():
    raw_data = pd.read_table('../../data/segmentation.test', sep=',', header=None, lineterminator='\n')

    x = raw_data.values[:, 6:]
    y = raw_data.values[:, 0]
    visualize_data(x, y)
    visualize_3d_data(x, y)

    data = Data(raw_data, 0.7)
    visualize_lda(data)
    test_lda(data)
    test_qda(data)
