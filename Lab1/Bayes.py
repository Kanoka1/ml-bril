import math

class Bayes():
    summaries = {}

    def summarize(self, items):
        summary = []
        length = len(items[0])
        i = 0
        while (i < length):
            values = []
            for item in items:
                values.append(item[i])
            summary.append([self.mean(values), self.stand_dev(values)])
            i += 1;
        return summary

    def devide_by_class(self, train_x, train_y):
        classes_dict = {}
        for item in zip(train_x, train_y):
            classes_dict.setdefault(item[1], [])
            classes_dict[item[1]].append(item[0])
        return classes_dict

    def fit(self, train_x, train_y):
        classes_dict = self.devide_by_class(train_x, train_y)
        for class_name, items in classes_dict.items():
            self.summaries[class_name] = self.summarize(items)

    def mean(self, values):
        return sum(values) / float(len(values))

    def stand_dev(self, values):
        var = sum((x - self.mean(values)) ** 2 for x in values) / float(len(values) - 1)
        return math.sqrt(var)

    def calc_probability(self, x, mean, stdev):
        if stdev == 0:
            stdev += 0.000001
        exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
        return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent

    def calc_class_probabilities(self, summaries, instance_attr):
        probabilities = {}
        for class_name, class_summaries in summaries.items():
            probabilities[class_name] = 1.0
            for i in range(len(class_summaries)):
                mean, stdev = class_summaries[i]
                x = float(instance_attr[i])
                probabilities[class_name] *= self.calc_probability(x, mean, stdev)
        return probabilities

    def predict_one(self, summaries, x):
        probabilities = self.calc_class_probabilities(summaries, x)
        best_class = None
        max_prob = -1
        for class_name, probability in probabilities.items():
            if best_class is None or probability > max_prob:
                max_prob = probability
                best_class = class_name
        return best_class

    def predict(self, data_x):
        if (self.summaries == None):
            raise Exception(
                "This NBCLassifier instance is not fitted yet. Call 'fit' with appropriate arguments before using this method.")
        predictions = []
        for x in data_x:
            predictions.append(self.predict_one(self.summaries, x))
        return predictions