from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier


class DecisionTree:
    def __init__(self, data):
        self.data = data
        self.dtc = DecisionTreeClassifier()
        self.rfc = RandomForestClassifier()

    def run(self):
        self.dtc.fit(self.data.train_x, self.data.train_y)
        self.rfc.fit(self.data.train_x, self.data.train_y)

        dtc_predict = self.dtc.predict(self.data.test_x)
        rfc_predict = self.rfc.predict(self.data.test_x)

        dtc_accuracy = accuracy_score(self.data.test_y, dtc_predict)
        rfc_accuracy = accuracy_score(self.data.test_y, rfc_predict)

        return dtc_accuracy, rfc_accuracy
