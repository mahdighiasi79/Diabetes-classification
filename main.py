import pandas as pd
# import matplotlib.pyplot as plt
# import preprocessing as prep
# import numpy as np
import helper_functions as hf
import pickle

if __name__ == "__main__":
    df = pd.read_csv("diabetic_data_without_missing_values.csv")
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

    # labels = df["readmitted"]
    # mi = {"race": hf.MutualInformation(df["race"], labels),
    #       "gender": hf.MutualInformation(df["gender"], labels),
    #       "admission_type_id": hf.MutualInformation(df["admission_type_id"], labels),
    #       "discharge_disposition_id": hf.MutualInformation(df["discharge_disposition_id"], labels),
    #       "admission_source_id": hf.MutualInformation(df["admission_source_id"], labels),
    #       "medical_specialty": hf.MutualInformation(df["medical_specialty"], labels),
    #       "diag_1": hf.MutualInformation(df["diag_1"], labels),
    #       "diag_2": hf.MutualInformation(df["diag_2"], labels),
    #       "diag_3": hf.MutualInformation(df["diag_3"], labels),
    #       "max_glu_serum": hf.MutualInformation(df["max_glu_serum"], labels),
    #       "A1Cresult": hf.MutualInformation(df["A1Cresult"], labels),
    #       "metformin": hf.MutualInformation(df["metformin"], labels),
    #       "repaglinide": hf.MutualInformation(df["repaglinide"], labels),
    #       "nateglinide": hf.MutualInformation(df["nateglinide"], labels),
    #       "chlorpropamide": hf.MutualInformation(df["chlorpropamide"], labels),
    #       "glimepiride": hf.MutualInformation(df["glimepiride"], labels),
    #       "glipizide": hf.MutualInformation(df["glipizide"], labels),
    #       "glyburide": hf.MutualInformation(df["tolbutamide"], labels),
    #       "pioglitazone": hf.MutualInformation(df["pioglitazone"], labels),
    #       "rosiglitazone": hf.MutualInformation(df["rosiglitazone"], labels),
    #       "acarbose": hf.MutualInformation(df["acarbose"], labels),
    #       "miglitol": hf.MutualInformation(df["miglitol"], labels),
    #       "tolazamide": hf.MutualInformation(df["tolazamide"], labels),
    #       "insulin": hf.MutualInformation(df["insulin"], labels),
    #       "glyburide-metformin": hf.MutualInformation(df["glyburide-metformin"], labels),
    #       "change": hf.MutualInformation(df["change"], labels),
    #       "diabetesMed": hf.MutualInformation(df["diabetesMed"], labels)}
    # with open("mutual_information.pkl", "wb") as file:
    #     pickle.dump(mi, file)
    # with open("mutual_information.pkl", "rb") as file:
    #     mi = pickle.load(file)
    #
    # sorted_mi = sorted(mi.items(), key=lambda x: x[1])
    # print(sorted_mi)

    values = {}
    for value in df["num_medications"]:
        if values.get(value) is None:
            values[value] = 1
        else:
            values[value] += 1
    print(values)
    print(len(values.keys()))

    print(hf.MutualInformation(df["num_lab_procedures"], df["readmitted"]))
