import helper_functions as hf
import pandas as pd
import numpy as np
import pickle

train_size = 80000


class NaiveBayesBinary:

    def __init__(self):
        self.df = pd.read_csv("preprocessed_data.csv")
        self.probabilities = {}
        with open("binary_labels.pkl", "rb") as file:
            self.labels = pickle.load(file)
            file.close()
        self.labels = np.array(self.labels)
        self.BayesianProbabilities()

    def CalculateProbability(self, attribute, value):
        feature = np.array(self.df[attribute][:train_size])
        condition = (feature == value)
        label0 = (self.labels[:train_size] == 0)
        label1 = (self.labels[:train_size] == 1)
        matches0 = condition * label0
        matches1 = condition * label1
        number0 = np.sum(matches0, axis=0, keepdims=False)
        number1 = np.sum(matches1, axis=0, keepdims=False)
        probability0 = number0 / train_size
        probability1 = number1 / train_size
        return [probability0, probability1]

    def BayesianProbabilities(self):
        for attribute in self.df.columns:
            probabilities = {}
            values = hf.ExtractValues(attribute)
            for value in values.keys():
                probabilities[value] = self.CalculateProbability(attribute, value)
            self.probabilities[attribute] = probabilities

    def Predict(self, record_id):
        record = self.df.iloc(0)[record_id]
        p0 = 1
        p1 = 1

        for attribute in self.df.columns:
            value = record[attribute]
            p0 *= self.probabilities[attribute][value][0]
            p1 *= self.probabilities[attribute][value][1]

        if p0 > p1:
            return 0
        else:
            return 1


if __name__ == "__main__":
    NB = NaiveBayesBinary()
    test_size = len(NB.df.iloc(0)[train_size:])
    true_predictions = 0
    for i in range(test_size):
        index = i + train_size
        prediction = NB.Predict(index)
        if prediction == NB.labels[index]:
            true_predictions += 1
    percentage = (true_predictions / test_size) * 100
    print(percentage)
