"""
Prior cleaning of duplicates. Reduces overhead
"""

import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv("../../Data/Controls_v3/Flowsheet_Vitals.txt")

    print(len(df))

    df = df.drop_duplicates(
        subset=[
            "IP_PATIENT_ID",
            "VITAL_SIGN_TYPE",
            "VITAL_SIGN_TAKEN_TIME",
            "VITAL_SIGN_VALUE",
        ]
    )

    print(len(df))

    df.to_csv("../../Data/Controls_v3/Flowsheet_Vitals_noduplicates.txt", index=False)
