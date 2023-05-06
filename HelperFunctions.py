import math
import numpy as np


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
