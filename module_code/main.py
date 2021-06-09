import logging
import sys
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
        # Keep a log of how preprocessing went. can call logger anywhere inside of logic from here
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s [%(levelname)s] %(message)s",
            # print to stdout and log to file
            handlers=[
                logging.FileHandler("dialysis_preproc.log"),
                logging.StreamHandler(sys.stdout),
            ],
        )
        logging.info("Preprocessed file does not exist! Creating...")
        start_time = time.time()
        df = merge_features_with_outcome()  # 140s ~2.5 mins
        logging.info(f"Loading took {time.time() - start_time} seconds.")
        df.to_feather(os.path.join(DATA_DIR, "combined_df.feather"))

    preprocessed_df = preprocess_data(df)
    scores = run_cv(preprocessed_df)
