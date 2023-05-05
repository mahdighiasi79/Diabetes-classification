import pandas as pd


if __name__ == "__main__":
    with open("diabetic_data.csv") as csv_file:
        file = csv.reader(csv_file)
        file = list(file)
        for rows in file:

