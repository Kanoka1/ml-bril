from sklearn.model_selection import train_test_split


class Data:
    def __init__(self, data, train_size):
        attributes = data.values[:, 4:]
        classes = data.values[:, 0]
        self.train_size = train_size
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(attributes, classes,
                                                                                train_size=train_size)
        self.y_names = set(classes)
