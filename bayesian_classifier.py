import helper_functions as hf
import pandas as pd
import numpy as np
import pickle
import math


class NaiveBayes:

    def __init__(self, attributes, labels):
        self.num_records = len(attributes)
        self.attributes = attributes
        self.labels = np.array(labels)
        self.probabilities = {}
        self.label0 = (self.labels == 0)
        self.label1 = (self.labels == 1)
        self.label2 = (self.labels == 2)
        self.num_label0 = np.sum(self.label0, axis=0, keepdims=False)
        self.num_label1 = np.sum(self.label1, axis=0, keepdims=False)
        self.num_label2 = np.sum(self.label2, axis=0, keepdims=False)
        self.BayesianProbabilities()

    def CalculateProbability(self, attribute, value):
        feature = np.array(self.attributes[attribute])
        condition = (feature == value)
        matches0 = condition * self.label0
        matches1 = condition * self.label1
        matches2 = condition * self.label2
        number0 = np.sum(matches0, axis=0, keepdims=False)
        number1 = np.sum(matches1, axis=0, keepdims=False)
        number2 = np.sum(matches2, axis=0, keepdims=False)
        probability0 = number0 / self.num_label0
        probability1 = number1 / self.num_label1
        probability2 = number2 / self.num_label2
        return [probability0, probability1, probability2]

    def BayesianProbabilities(self):
        for attribute in self.attributes.columns:
            probabilities = {}
            values = hf.ExtractValues(attribute)
            for value in values.keys():
                probabilities[value] = self.CalculateProbability(attribute, value)
            self.probabilities[attribute] = probabilities

        p_class0 = self.num_label0 / self.num_records
        p_class1 = self.num_label1 / self.num_records
        p_class2 = self.num_label2 / self.num_records
        self.probabilities["readmitted"] = [p_class0, p_class1, p_class2]

    def Predict(self, record):
        p0 = 1
        p1 = 1
        p2 = 1

        for attribute in self.attributes.columns:
            value = record[attribute]
            p0 *= self.probabilities[attribute][value][0]
            p1 *= self.probabilities[attribute][value][1]
            p2 *= self.probabilities[attribute][value][2]

        p0 *= self.probabilities["readmitted"][0]
        p1 *= self.probabilities["readmitted"][1]
        p2 *= self.probabilities["readmitted"][2]

        return np.argmax(np.array([p0, p1, p2]))


if __name__ == "__main__":
    df = pd.read_csv("preprocessed_data.csv")
    records = len(df)
    with open("labels.pkl", "rb") as file:
        converted_labels = pickle.load(file)
        file.close()

    train_size = math.floor(records * 0.9)
    test_size = records - train_size
    train_set = df.iloc(0)[:train_size]
    train_set = train_set.drop("readmitted", axis=1)

    NB = NaiveBayes(train_set, converted_labels[:train_size])

    true_predictions = 0
    for i in range(test_size):
        index = i + train_size
        prediction = NB.Predict(df.iloc(0)[index])
        if prediction == converted_labels[index]:
            true_predictions += 1
    accuracy = (true_predictions / test_size) * 100
    print("accuracy:", accuracy)
