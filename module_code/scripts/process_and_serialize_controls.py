from argparse import Namespace
from timeit import default_timer as timer
from logging import info
from typing import List, Optional, Dict
import sys
from os.path import join
from os import getcwd

from functools import reduce
from pandas import DataFrame, DatetimeIndex, merge

sys.path.insert(0, join(getcwd(), "module_code"))

from cli_utils import load_cli_args, init_cli_args
from data.longitudinal_features import load_procedures
from data.load import merge_features_with_outcome
from data.utils import time_delta_to_str


def construct_outcomes(procedures_df: DataFrame, merge_on: List[str]) -> DataFrame:
    """
    Instead of CRRT start date we use a basic heuristic for a "anchoring" date / pseudo start date for time windows
    # In controls this is 4725 patients
    """
    outcomes = (
        procedures_df.groupby("IP_PATIENT_ID")
        # We randomly select the date of a procedure for any patient
        .sample(n=1)[["IP_PATIENT_ID", "PROC_DATE"]]
        # Ensure date is a datetime object
        .astype({"PROC_DATE": "datetime64[ns]"}).assign(
            **{
                # they should not go on crrt and did not
                "recommend_crrt": 0,
                # because they never went on crrt theres 0 days and no prev treatments
                "Days on CRRT": 0,
                "Num Prev CRRT Treatments": 0,
                # TODO: does it make sense to have this anymore?
                # crrt_year=lambda row: DatetimeIndex(row["PROCEDURE_DATE"]).year,
                "CRRT Year": lambda df: df["PROC_DATE"].map(lambda dt: dt.year),
                # need this for get_time_window_mask, will not be used
                "End Date": lambda df: df["PROC_DATE"],
            }
        )
        # Align column names to a real outcomes file
        .rename(columns={"PROC_DATE": "Start Date"})
    )

    return outcomes.set_index(merge_on)


if __name__ == "__main__":
    load_cli_args()
    args = init_cli_args()

    controls_window_size = args.pre_start_delta

    slide_window_by = f"+{args.slide_window_by}" if args.slide_window_by else ""
    fname = f"df_controls_{time_delta_to_str(controls_window_size)}{slide_window_by}.{args.serialization}"

    info(f"Creating preprocessed file {fname}...")
    start = timer()
    merge_on = ["IP_PATIENT_ID", "Start Date"]

    procedures_df = load_procedures(args.raw_data_dir, aggregate=False)
    outcomes_df = construct_outcomes(procedures_df, merge_on)
    df = merge_features_with_outcome(
        args.raw_data_dir,
        outcomes_df,
        merge_on,
        args.time_interval,
        controls_window_size,
        None,
        "Start Date",
        args.slide_window_by,
    )
    info(f"Took {timer() - start} seconds.")

    serialize_fn = getattr(df, f"to_{args.serialization}")
    serialize_fn(join(args.raw_data_dir, fname))
