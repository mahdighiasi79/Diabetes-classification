import pandas as pd
import preprocessing as pre


if __name__ == "__main__":
    df = pd.read_csv("preprocessed_data.csv")
    print(df.columns)
    print(len(df))
