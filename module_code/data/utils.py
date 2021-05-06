import os
import pandas as pd
from functools import reduce
from typing import List

DATA_DIR = "/home/davina/Private/dialysis-data"


def read_files_and_combine(
    files: List[str], on: List[str] = ["IP_PATIENT_ID"], how: str = "inner"
) -> pd.DataFrame:
    """
    Takes one or more files in a list and returns a combined dataframe.
    Deals with strangely formatted files from the dataset.
    """
    dfs = []

    for file in files:
        try:
            # Try normally reading the csv with pandas, if it fails the formatting is strange
            dfs.append(pd.read_csv(f"{DATA_DIR}/{file}"))
        except:
            print(f"Unexpected encoding in {file}")
            default_guess = "cp1252"

            # get file encoding using file -i and extracting name with sed
            # ref: https://unix.stackexchange.com/a/393949
            # -n: don't print unless we say. s/ search, .* match any, charset=, // remove text up until after =, print remaining
            command = f"file -i {DATA_DIR}/{file} | sed -n 's/.*charset=//p'"
            # [:-1] ignore newline
            encoding = os.popen(command).read()[:-1]
            print(f"Encoding was {encoding} instead of assumed utf-8.")

            # Try reading the file with the assumed or inferred encoding.
            if encoding == "unknown-8bit":
                print(f"Assuming {default_guess}...")
                dfs.append(pd.read_csv(f"{DATA_DIR}/{file}", encoding=default_guess))
            else:
                dfs.append(pd.read_csv(f"{DATA_DIR}/{file}", encoding=encoding))

    combined = reduce(lambda df1, df2: pd.merge(df1, df2, on=on, how=how), dfs)
    return combined
