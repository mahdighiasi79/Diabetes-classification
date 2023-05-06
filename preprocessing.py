import pandas as pd
import math
import numpy as np


def EliminateMissingValues():
    df = pd.read_csv("diabetic_data.csv")
    records = len(df)
    drops = []

    # for race attribute, the most frequent value is replaced with missing values
    races = df["race"]
    for i in range(records):
        if races[i] == '?':
            df.iloc[i, 2] = "Caucasian"

    # for gender attribute, the rows with missing values are deleted
    genders = df["gender"]
    for i in range(records):
        if genders[i] == '?':
            if i not in drops:
                drops.append(i)

    # for age attribute, the median of the interval is replaced with the intervals
    ages = df["age"]
    for i in range(records):
        if ages[i] == "[0-10)":
            df.iloc[i, 4] = 5
        elif ages[i] == "[10-20)":
            df.iloc[i, 4] = 15
        elif ages[i] == "[20-30)":
            df.iloc[i, 4] = 25
        elif ages[i] == "[30-40)":
            df.iloc[i, 4] = 35
        elif ages[i] == "[40-50)":
            df.iloc[i, 4] = 45
        elif ages[i] == "[50-60)":
            df.iloc[i, 4] = 55
        elif ages[i] == "[60-70)":
            df.iloc[i, 4] = 65
        elif ages[i] == "[70-80)":
            df.iloc[i, 4] = 75
        elif ages[i] == "[80-90)":
            df.iloc[i, 4] = 85
        elif ages[i] == "[90-100)":
            df.iloc[i, 4] = 95

    # for weight attribute, the mean of the existing values is replaced by missing values
    # and the median of intervals will be replaced by the intervals
    weights = df["weight"]
    for i in range(records):
        if weights[i] == "[0-25)":
            df.iloc[i, 5] = 12.5
        elif weights[i] == "[25-50)":
            df.iloc[i, 5] = 37.5
        elif weights[i] == "[50-75)":
            df.iloc[i, 5] = 62.5
        elif weights[i] == "[75-100)":
            df.iloc[i, 5] = 87.5
        elif weights[i] == "[100-125)":
            df.iloc[i, 5] = 112.5
        elif weights[i] == "[125-150)":
            df.iloc[i, 5] = 137.5
        elif weights[i] == "[150-175)":
            df.iloc[i, 5] = 162.5
        elif weights[i] == "[175-200)":
            df.iloc[i, 5] = 187.5
        elif weights[i] == ">200":
            df.iloc[i, 5] = 212.5
        else:
            df.iloc[i, 5] = 86.28010634970285

    # for diag_1, diag_2, and diag_3 attribute, the rows with missing values are deleted
    diag_1 = df["diag_1"]
    for i in range(records):
        if diag_1[i] == '?':
            if i not in drops:
                drops.append(i)

    diag_2 = df["diag_2"]
    for i in range(records):
        if diag_2[i] == '?':
            if i not in drops:
                drops.append(i)

    diag_3 = df["diag_3"]
    for i in range(records):
        if diag_3[i] == '?':
            if i not in drops:
                drops.append(i)

    df.drop(drops, inplace=True)
    df.to_csv("diabetic_data_without_missing_values.csv")


def normalize(feature):
    records = len(feature)

    mean = 0
    for value in feature:
        mean += value
    mean /= records

    variance = 0
    for value in feature:
        variance += pow(value - mean, 2)
    variance /= records - 1
    standard_deviation = pow(variance, 0.5)

    result = []
    for value in feature:
        value -= mean
        value /= standard_deviation
        result.append(value)
    return result


def Entropy(feature):
    records = len(feature)
    values = {}

    for value in feature:
        if values.get(value) is None:
            values[value] = 1
        else:
            values[value] += 1

    entropy = 0
    for key in values.keys():
        probability = values[key] / records
        entropy += probability * math.log2(probability)
    entropy *= -1
    return entropy


def MutualInformation(feature1, feature2):
    records = len(feature1)
    outcomes1 = []
    outcomes2 = []

    for value in feature1:
        if value not in outcomes1:
            outcomes1.append(value)

    for value in feature2:
        if value not in outcomes2:
            outcomes2.append(value)

    n1 = len(outcomes1)
    n2 = len(outcomes2)
    probability_matrix = np.zeros((n1, n2))

    for value1 in feature1:
        for value2 in feature2:
            row = outcomes1.index(value1)
            column = outcomes2.index(value2)
            probability_matrix[row][column] += 1
    probability_matrix /= records

    joint_entropy = 0
    for i in range(n1):
        for j in range(n2):
            probability = probability_matrix[i][j]
            joint_entropy += probability * math.log2(probability)
    joint_entropy *= -1
    
    mutual_information = Entropy(feature1) + Entropy(feature2) - joint_entropy
    return mutual_information
