from data.utils import DATA_DIR
from data.load import load_outcomes, merge_features_with_outcome, load_problems
import pandas as pd
import time

if __name__ == "__main__":
    try:
        # raise IOError
        df = pd.read_feather(f"{DATA_DIR}/combined_df.feather")
    except IOError:
        print("Preprocessed file does not exist! Creating...")
        start_time = time.time()
        df = merge_features_with_outcome()  # 140s ~2.5 mins
        print(f"Loading took {time.time() - start_time} seconds.")
        df.to_feather(f"{DATA_DIR}/combined_df.feather")
