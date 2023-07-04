from argparse import ArgumentParser
from os.path import join
from os import getcwd
import sys

sys.path.insert(0, join(getcwd(), "module_code"))

from cli_utils import load_cli_args, init_cli_args
from data.load import (
    load_static_features,
    process_and_serialize_raw_data,
    get_preprocessed_df_path,
)
from exp.utils import time_delta_str_to_dict


def main():
    load_cli_args()
    p = ArgumentParser()
    p.add_argument(
        "--cohort",
        type=str,
        choices=["ucla_crrt", "cedars_crrt", "ucla_control"],
        help="Name of cohort to run preprocessing on.",
    )
    p.add_argument(
        "--str-pre-start-delta",
        type=str,
        help="Convenient string version of pre-start-delta. Will be converted to dict",
    )
    p.add_argument(
        "--reverse",
        action="store_true",
        help="If true, reverses the direction of the rolling window",
    )
    args = init_cli_args(p)

    args.pre_start_delta = time_delta_str_to_dict(args.str_pre_start_delta)

    if args.reverse:
        args.slide_window_by = args.slide_window_by * -1

    process_and_serialize_raw_data(
        args, get_preprocessed_df_path(args, args.cohort), args.cohort
    )

    # Also do static data.
    cohort_data_dir = getattr(args, f"{args.cohort}_data_dir")
    path = join(cohort_data_dir, f"static_data.{args.serialization}")
    static_features = load_static_features(cohort_data_dir).set_index("IP_PATIENT_ID")
    serialize_fn = getattr(static_features, f"to_{args.serialization}")
    serialize_fn(path)


if __name__ == "__main__":
    """
    Run with:
        `python module_code/scripts/process_and_serialize_raw_data.py  --cohort ucla_control`
    For the rolling window:
    ```
    for i in {0..7}; do python module_code/scripts/process_and_serialize_raw_data.py  --cohort ucla_crrt --slide-window-by $i; done
    ```
    To run rolling window in parallel:
    ```
    for i in {0..7}; do python module_code/scripts/process_and_serialize_raw_data.py  --cohort ucla_crrt --slide-window-by $i & done; wait;
    ```

    For all pre-start-delta required for hparam tuning:
    Option 1: All cohorts at once
    ```
    for j in {cedars_crrt,ucla_crrt,ucla_control}; do for i in {1,2,3,4,5,6,7,10,14}; do python module_code/scripts/process_and_serialize_raw_data.py  --cohort ${j} --slide-window-by 0 --str-pre-start-delta ${i}d & done; done; wait;
    ```
    Option 2: Cohorts one-by-one
    ```
    for i in {1,2,3,4,5,6,7,10,14}; do python module_code/scripts/process_and_serialize_raw_data.py  --cohort cedars_crrt --slide-window-by 0 --str-pre-start-delta ${i}d & done; wait;
    for i in {1,2,3,4,5,6,7,10,14}; do python module_code/scripts/process_and_serialize_raw_data.py  --cohort ucla_crrt --slide-window-by 0 --str-pre-start-delta ${i}d & done; wait;
    for i in {1,2,3,4,5,6,7,10,14}; do python module_code/scripts/process_and_serialize_raw_data.py  --cohort ucla_control --slide-window-by 0 --str-pre-start-delta ${i}d & done; wait;

    Example of rolling back instead of forwards:
    Option 1: all cohorts at once
    ```
    for j in {cedars_crrt,ucla_crrt,ucla_control}; do for i in {1,2,3}; do python module_code/scripts/process_and_serialize_raw_data.py  --reverse --cohort ${j} --slide-window-by ${i} --str-pre-start-delta 1d & done; done; wait;
    ```
    Option 2: Cohorts one-by-one
    ```
    for i in {1,2,3}; do python module_code/scripts/process_and_serialize_raw_data.py  --invert --cohort cedars_crrt --slide-window-by ${i} --str-pre-start-delta 1d & done; wait;
    for i in {1,2,3}; do python module_code/scripts/process_and_serialize_raw_data.py  --invert --cohort ucla_crrt --slide-window-by ${i}  --str-pre-start-delta 1d & done; wait;
    for i in {1,2,3}; do python module_code/scripts/process_and_serialize_raw_data.py  --invert --cohort ucla_control --slide-window-by ${i} --str-pre-start-delta 1d & done; wait;
    ```

    """
    # sys.argv += ["--cohort", "cedars_crrt"]
    # sys.argv += ["--cohort", "ucla_crrt"]
    # sys.argv += ["--cohort", "ucla_control"]
    # sys.argv += ["--slide-window-by", "0"]
    main()
