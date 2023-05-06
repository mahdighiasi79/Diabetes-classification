import pandas as pd
import matplotlib.pyplot as plt
import preprocessing
import numpy as np


if __name__ == "__main__":
    # df = pd.read_csv("diabetic_data_without_missing_values.csv")
    # records = len(df)
    # feature1 = df["weight"]
    # feature2 = df["num_medications"]
    # labels = df["readmitted"]
    # feature1 = preprocessing.normalize(feature1)
    # feature2 = preprocessing.normalize(feature2)
    #
    # x = []
    # y = []
    # for i in range(records):
    #     if labels[i] == "NO":
    #         x.append(feature1[i])
    #         y.append(feature2[i])
    # plt.scatter(x, y, color="green")
    #
    # x = []
    # y = []
    # for i in range(records):
    #     if labels[i] == ">30":
    #         x.append(feature1[i])
    #         y.append(feature2[i])
    # plt.scatter(x, y, color="red")
    #
    # x = []
    # y = []
    # for i in range(records):
    #     if labels[i] == "<30":
    #         x.append(feature1[i])
    #         y.append(feature2[i])
    # plt.scatter(x, y, color="black")
    #
    # plt.show()

    # values = {}
    # labels = df["readmitted"]
    # for value in labels:
    #     if values.get(value) is None:
    #         values[value] = 1
    #     else:
    #         values[value] += 1
    # print(values)

    arr = [1, 2, 3]
    index = arr.index(3)
    print(index)
