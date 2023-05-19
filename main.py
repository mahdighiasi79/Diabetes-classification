import pandas as pd


if __name__ == "__main__":
    df = pd.read_csv("preprocessed_data.csv")
    row = df.iloc(0)[1]
    print(row)
    print(row["diag_1"])
