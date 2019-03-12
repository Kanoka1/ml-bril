import pandas as pnd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


class Data:
    def __init__(self, data, train_size):
        self.x = data.values[:, 1:]
        self.y = data.values[:, 0]
        self.train_size = train_size
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(self.x, self.y,
                                                                                train_size=train_size)
        self.y_names = set(self.y)


def calculate_metrics(data, gnb, tree):
    scoring = ['accuracy', 'neg_log_loss', 'roc_auc']

    gnb_scores = cross_validate(gnb, data.x, data.y, scoring=scoring,
                                cv=5, return_train_score=False)

    tree_scores = cross_validate(tree, data.x, data.y, scoring=scoring,
                                 cv=5, return_train_score=False)

    gnb_mean, gnb_std = gnb_scores['test_accuracy'].mean(), gnb_scores['test_accuracy'].std()
    tree_mean, tree_std = tree_scores['test_accuracy'].mean(), tree_scores['test_accuracy'].std()
    print('Accuracy for GaussianNB: %0.3f (+/- %0.2f)\n'
          'Accuracy for RandomTree: %0.3f (+/- %0.2f)\n' % (gnb_mean, gnb_std, tree_mean, tree_std))

    gnb_mean, gnb_std = gnb_scores['test_neg_log_loss'].mean(), gnb_scores['test_neg_log_loss'].std()
    tree_mean, tree_std = tree_scores['test_neg_log_loss'].mean(), tree_scores['test_neg_log_loss'].std()
    print('Logarithmic loss for GaussianNB: %0.3f (+/- %0.2f)\n'
          'Logarithmic loss for RandomTree: %0.3f (+/- %0.2f)\n' % (gnb_mean, gnb_std, tree_mean, tree_std))

    gnb_mean, gnb_std = gnb_scores['test_roc_auc'].mean(), gnb_scores['test_roc_auc'].std()
    tree_mean, tree_std = tree_scores['test_roc_auc'].mean(), tree_scores['test_roc_auc'].std()
    print('Area under ROC curve for GaussianNB: %0.3f (+/- %0.2f)\n'
          'Area under ROC curve for RandomTree: %0.3f (+/- %0.3f)\n' % (gnb_mean, gnb_std, tree_mean, tree_std))


def confus_matrix(data, gnb, tree):
    gnb_matrix = confusion_matrix(data.test_y, gnb.predict(data.test_x))
    print("Confusion matrix for GaussianNB")
    print(gnb_matrix)

    tree_matrix = confusion_matrix(data.test_y, tree.predict(data.test_x))
    print("Confusion matrix for RandomForest")
    print(tree_matrix)


def class_report(data, gnb, tree):
    gnb_report = classification_report(data.test_y, gnb.predict(data.test_x))
    print('Classification report for GaussianNB:')
    print(gnb_report)

    lda_report = classification_report(data.test_y, tree.predict(data.test_x))
    print('Classification report for RandomForest:')
    print(lda_report)


def get_data():
    raw_data = pnd.read_table('../../data/spectf.data', sep=',', header=None, lineterminator='\n')
    return Data(raw_data, 0.7)


def main():
    data = get_data()

    gnb = GaussianNB()
    tree = RandomForestClassifier()

    _ = gnb.fit(data.train_x, data.train_y)
    _ = tree.fit(data.train_x, data.train_y)
    calculate_metrics(data, gnb, tree)
    confus_matrix(data, gnb, tree)
    class_report(data, gnb, tree)


main()
