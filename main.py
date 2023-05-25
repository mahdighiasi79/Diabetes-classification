import preprocessing as pre
import pandas as pd

if __name__ == "__main__":
    # pre.EliminateMissingValues()
    # pre.FeatureSelection()
    # pre.EliminateOutliers()
    # pre.ConvertLabels()

    # database: https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008
    # article: https://www.hindawi.com/journals/bmri/2014/781670/#materials-and-methods

    arr1 = [1, 2, 3, 4, 5, 6, 7]
    arr2 = [2, 4, 5]
    s1 = set(arr1)
    s2 = set(arr2)
    print(s1.difference(s2))
    print(list(s1.difference(s2)))

    print("hello")
