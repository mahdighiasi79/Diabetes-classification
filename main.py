import pandas as pd


if __name__ == "__main__":
    df = pd.read_csv("diabetic_data.csv")
    print(df.iloc(0)[0])
