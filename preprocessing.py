import pickle
import numpy as np
import pandas as pd
import helper_functions as hf


selected_features = ["admission_type_id", "discharge_disposition_id", "admission_source_id", "time_in_hospital", "medical_specialty",
                     "num_lab_procedures", "num_medications", "number_outpatient", "number_emergency", "number_inpatient", "diag_1", "diag_2",
                     "diag_3", "number_diagnoses", "tolbutamide", "insulin", "change", "diabetesMed", "readmitted"]

selected_features_categorical = ["admission_type_id", "discharge_disposition_id", "admission_source_id", "medical_specialty", "diag_1",
                                 "diag_2", "diag_3", "tolbutamide", "insulin", "change", "diabetesMed"]

selected_features_numerical = ["time_in_hospital", "num_lab_procedures", "num_medications", "number_outpatient", "number_emergency",
                               "number_inpatient", "number_diagnoses"]


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


def FeatureSelection():
    df = pd.read_csv("diabetic_data_without_missing_values.csv")
    for column in df.columns:
        if column not in selected_features:
            df.drop(column, axis=1, inplace=True)
    df.to_csv("selected_features.csv")


def EliminateOutliers():
    outliers = []
    df = pd.read_csv("selected_features.csv")

    for feature in selected_features_categorical:
        noises = hf.DetectOutliersCategorical(df[feature])
        outliers = list(set(outliers) | set(noises))

    for feature in selected_features_numerical:
        answers = hf.DetectOutliersNumerical(df[feature])
        noises = []
        for i in range(len(answers)):
            if answers[i]:
                noises.append(i)
        outliers = list(set(outliers) | set(noises))

    df.drop(outliers, axis=0, inplace=True)
    df.to_csv("preprocessed_data.csv")


def PrepareANNInput():
    df = pd.read_csv("preprocessed_data.csv")
    data = [df["time_in_hospital"], df["num_lab_procedures"], df["num_medications"], df["number_outpatient"], df["number_emergency"],
            df["number_inpatient"], df["number_diagnoses"]]
    rows = len(data)
    columns = len(data[0])

    ann_input = []
    for i in range(rows):
        row = []
        for j in range(columns):
            row.append(int(data[i][j]))
        ann_input.append(row)
    ann_input = np.array(ann_input)
    ann_input = np.transpose(ann_input)

    for i in range(len(ann_input)):
        ann_input[i] = hf.Normalize(ann_input[i])

    with open("ann_input.pkl", "wb") as file:
        pickle.dump(ann_input, file)
        file.close()


def PrepareANNLabels():
    df = pd.read_csv("preprocessed_data.csv")
    readmitted = df["readmitted"]
    labels = []
    for i in range(len(readmitted)):
        if readmitted[i] == "NO":
            labels.append([1, 0, 0])
        elif readmitted[i] == ">30":
            labels.append([0, 1, 0])
        else:
            labels.append([0, 0, 1])

    with open("ann_labels.pkl", "wb") as file:
        pickle.dump(labels, file)
        file.close()
