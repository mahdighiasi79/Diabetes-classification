import helper_functions as hf
import preprocessing as pre
import pandas as pd
import numpy as np
import pickle

train_size = 80000


class NaiveBayes:

    def __init__(self):
        self.df = pd.read_csv("preprocessed_data.csv")
        self.probabilities = {}
        with open("labels.pkl", "rb") as file:
            self.labels = pickle.load(file)
            file.close()
        self.labels = np.array(self.labels)
        self.BayesianProbabilities()

    def CalculateProbability(self, attribute, value):
        feature = np.array(self.df[attribute][:train_size])
        label0 = (self.labels[:train_size] == 0)
        label1 = (self.labels[:train_size] == 1)
        label2 = (self.labels[:train_size] == 2)
        if attribute in pre.selected_features_categorical:
            condition = (feature == value)
            matches0 = condition * label0
            matches1 = condition * label1
            matches2 = condition * label2
            number0 = np.sum(matches0, axis=0, keepdims=False)
            number1 = np.sum(matches1, axis=0, keepdims=False)
            number2 = np.sum(matches2, axis=0, keepdims=False)
            probability0 = number0 / train_size
            probability1 = number1 / train_size
            probability2 = number2 / train_size
            return [probability0, probability1, probability2]
        elif attribute in pre.selected_features_numerical:
            feature = hf.ConvertNumericalFeatures(feature)
            matches0 = hf.EliminateZeros(feature, label0)
            matches1 = hf.EliminateZeros(feature, label1)
            matches2 = hf.EliminateZeros(feature, label2)
            mean0, variance0 = hf.MeanVariance(matches0)
            mean1, variance1 = hf.MeanVariance(matches1)
            mean2, variance2 = hf.MeanVariance(matches2)
            return [[mean0, variance0], [mean1, variance1], [mean2, variance2]]

    def BayesianProbabilities(self):
        for attribute in self.df.columns:

            if attribute in pre.selected_features_categorical:
                probabilities = {}
                values = hf.ExtractValues(attribute)
                for value in values.keys():
                    probabilities[value] = self.CalculateProbability(attribute, value)
                self.probabilities[attribute] = probabilities

            elif attribute in pre.selected_features_numerical:
                self.probabilities[attribute] = self.CalculateProbability(attribute, None)

    def Predict(self, record_id):
        record = self.df.iloc(0)[record_id]
        p0 = 1
        p1 = 1
        p2 = 1

        for attribute in self.df.columns:
            value = record[attribute]

            if attribute in pre.selected_features_categorical:
                p0 *= self.probabilities[attribute][value][0]
                p1 *= self.probabilities[attribute][value][1]
                p2 *= self.probabilities[attribute][value][2]
            elif attribute in pre.selected_features_numerical:
                mean0 = self.probabilities[attribute][0][0]
                mean1 = self.probabilities[attribute][1][0]
                mean2 = self.probabilities[attribute][2][0]
                variance0 = self.probabilities[attribute][0][1]
                variance1 = self.probabilities[attribute][0][1]
                variance2 = self.probabilities[attribute][0][1]
                p0 *= hf.NormalDistribution(value, mean0, variance0)
                p1 *= hf.NormalDistribution(value, mean1, variance1)
                p2 *= hf.NormalDistribution(value, mean2, variance2)

        return np.argmax(np.array([p0, p1, p2]))


if __name__ == "__main__":
    NB = NaiveBayes()
    test_size = len(NB.df.iloc(0)[train_size:])
    true_predictions = 0
    for i in range(test_size):
        index = i + train_size
        prediction = NB.Predict(index)
        if prediction == NB.labels[index]:
            true_predictions += 1
    percentage = (true_predictions / test_size) * 100
    print(percentage)
