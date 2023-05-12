import numpy as np
import pandas as pd
import pickle

df = pd.read_csv("preprocessed_data.csv")


def ConvertLabels(node):
    labels = []
    with open("labels.pkl", "rb") as file:
        readmitted = pickle.load(file)
        file.close()
    for record in node:
        labels.append(readmitted[record])
    return np.array(labels)


def GINI(node):
    records = len(node)
    labels = ConvertLabels(node)
    class_1 = (labels == [1, 0, 0])
    class_2 = (labels == [0, 1, 0])
    class_3 = (labels == [0, 0, 1])
    p1 = np.sum(class_1, axis=1, keepdims=False)
    p2 = np.sum(class_2, axis=1, keepdims=False)
    p3 = np.sum(class_3, axis=1, keepdims=False)
    p1 = np.sum(p1, axis=0, keepdims=False) / 3
    p2 = np.sum(p2, axis=0, keepdims=False) / 3
    p3 = np.sum(p3, axis=0, keepdims=False) / 3
    p1 /= records
    p2 /= records
    p3 /= records
    gini = 1 - (pow(p1, 2) + pow(p2, 2) + pow(p3, 2))
    return gini


def GINI_split(splits):
    gini_split = 0
    n = 0
    for split in splits:
        n_split = len(split)
        gini_split += n_split * GINI(split)
        n += n_split
    gini_split /= n
    return gini_split


def Split(parent, attribute, sets):
    feature = df[attribute]
    splits = []

    for s in sets:
        split = []

        for i in range(len(parent)):
            if feature[parent[i]] in s:
                split.append(parent[i])

        splits.append(split)

    return splits
