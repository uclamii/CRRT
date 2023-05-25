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


def main():
    load_cli_args()
    p = ArgumentParser()
    p.add_argument(
        "--cohort",
        type=str,
        choices=["ucla_crrt", "cedars_crrt", "ucla_control"],
        help="Name of cohort to run preprocessing on.",
    )
    args = init_cli_args(p)

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
    for i in {1..7}; do python module_code/scripts/process_and_serialize_raw_data.py  --cohort ucla_crrt --slide-window-by $i; done
    ```
    """
    # sys.argv += ["--cohort", "cedars_crrt"]
    main()
