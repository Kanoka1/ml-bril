from collections import Counter

import math


class KNN:
    def __init__(self, neighbours_size):
        self.neighbours_size = neighbours_size

    def fit(self, train_x, train_y):
        self.train_x = train_x
        self.train_y = train_y

    def get_euclidean_distance(self, inst1, inst2):
        distances = [(i - j) ** 2 for i, j in zip(inst1, inst2)]
        return math.sqrt(sum(distances))

    def get_neighbours(self, inst):
        distances = []
        for i in self.train_x:
            distances.append(self.get_euclidean_distance(i, inst))
        distances = tuple(zip(distances, self.train_y))
        sorted_list = sorted(distances)[:self.neighbours_size]

        class_name_list = []
        for value, class_name in sorted_list:
            class_name_list.append(class_name)
        return class_name_list

    def predict_class(self, neighbours):
        return Counter(neighbours).most_common()[0][0]

    def predict(self, data_x):
        predictions = []
        for x in data_x:
            neighbours = self.get_neighbours(x)
            predicted_class = self.predict_class(neighbours)
            predictions.append(predicted_class)
        return predictions
