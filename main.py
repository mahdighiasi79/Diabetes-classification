import pandas as pd
import matplotlib.pyplot as plt


if __name__ == "__main__":
    df = pd.read_csv("diabetic_data_without_missing_values.csv")
    feature1 = df["num_procedures"]
    feature2 = df["num_medications"]
    labels = df["readmitted"]
    
