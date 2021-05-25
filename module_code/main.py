from data.utils import DATA_DIR
from data.preprocess import preprocess_data
from data.load import merge_features_with_outcome
from exp.cv import run_cv
import pandas as pd
import time
import os

if __name__ == "__main__":
    try:
        # raise IOError
        df = pd.read_feather(os.path.join(DATA_DIR, "combined_df.feather"))
    except IOError:
        print("Preprocessed file does not exist! Creating...")
        start_time = time.time()
        df = merge_features_with_outcome()  # 140s ~2.5 mins
        print(f"Loading took {time.time() - start_time} seconds.")
        df.to_feather(os.path.join(DATA_DIR, "combined_df.feather"))

    preprocessed_df = preprocess_data(df)
    scores = run_cv(preprocessed_df)

