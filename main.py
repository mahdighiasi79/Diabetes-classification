import pandas as pd
import preprocessing as pre
import numpy as np


def ExtractValues(attribute):
    df = pd.read_csv("diabetic_data.csv")
    feature = df[attribute]
    values = {}
    for value in feature:
        if values.get(value) is None:
            values[value] = 1
        else:
            values[value] += 1
    return values


if __name__ == "__main__":
    arr = np.array([1, 2, 4])
    arr += [3]
    print(arr)
