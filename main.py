import helper_functions as hf

temp = ["250", "401", "272", "585", "403", "536", "278", "276", "305", "V10", "198", "414", "250.6"]


if __name__ == "__main__":
    diag2 = hf.ExtractValues("diag_3")
    values = []
    for key in diag2.keys():
        if key not in temp:
            values.append(key)
    print(values)
    # hf.InformativeValues("diag_3")
