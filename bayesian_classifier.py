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
        self.BayesianProbabilities()

    def CalculateProbability(self, attribute, value):
        feature = np.array(self.attributes[attribute])
        condition = (feature == value)
        label0 = (self.labels == 0)
        label1 = (self.labels == 1)
        label2 = (self.labels == 2)
        matches0 = condition * label0
        matches1 = condition * label1
        matches2 = condition * label2
        number0 = np.sum(matches0, axis=0, keepdims=False)
        number1 = np.sum(matches1, axis=0, keepdims=False)
        number2 = np.sum(matches2, axis=0, keepdims=False)
        probability0 = number0 / self.num_records
        probability1 = number1 / self.num_records
        probability2 = number2 / self.num_records
        return [probability0, probability1, probability2]

    def BayesianProbabilities(self):
        for attribute in self.attributes.columns:
            probabilities = {}
            values = hf.ExtractValues(attribute)
            for value in values.keys():
                probabilities[value] = self.CalculateProbability(attribute, value)
            self.probabilities[attribute] = probabilities

    def Predict(self, record):
        p0 = 1
        p1 = 1
        p2 = 1

        for attribute in self.attributes.columns:
            value = record[attribute]
            p0 *= self.probabilities[attribute][value][0]
            p1 *= self.probabilities[attribute][value][1]
            p2 *= self.probabilities[attribute][value][2]

        return np.argmax(np.array([p0, p1, p2]))


if __name__ == "__main__":
    df = pd.read_csv("preprocessed_data.csv")
    records = len(df)
    with open("labels.pkl", "rb") as file:
        converted_labels = pickle.load(file)
        file.close()

    overall_accuracy = 0

    ids = []
    for i in range(records):
        ids.append(i)

    train_size = math.floor(records * 0.9)
    test_size = records - train_size
    num_iterations = math.floor(records / test_size)
    for i in range(num_iterations):
        test_ids = ids[i * test_size: (i + 1) * test_size]
        train_set = df.drop(test_ids)
        train_labels = converted_labels[:i * test_size] + converted_labels[(i + 1) * test_size:]
        train_ids = list(set(ids).difference(set(test_ids)))
        test_set = df.drop(train_ids)
        test_labels = converted_labels[i * test_size: (i + 1) * test_size]

        NB = NaiveBayes(train_set, train_labels)

        true_predictions = 0
        for j in range(test_size):
            prediction = NB.Predict(test_set.iloc(0)[j])
            if prediction == test_labels[j]:
                true_predictions += 1
        percentage = (true_predictions / test_size) * 100
        overall_accuracy += percentage
        print(percentage)

    NB = NaiveBayes(df.iloc(0)[:train_size], converted_labels[:train_size])
    true_predictions = 0
    for i in range(test_size):
        index = i + train_size
        prediction = NB.Predict(df.iloc(0)[index])
        if prediction == converted_labels[index]:
            true_predictions += 1
    percentage = (true_predictions / test_size) * 100
    overall_accuracy += percentage
    print(percentage)

    overall_accuracy /= (num_iterations + 1)
    print(overall_accuracy)
