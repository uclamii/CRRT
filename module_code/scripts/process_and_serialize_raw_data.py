from os.path import join
from os import getcwd
import sys

sys.path.insert(0, join(getcwd(), "module_code"))

from cli_utils import load_cli_args, init_cli_args
from data.load import (
    process_and_serialize_raw_data,
    get_preprocessed_df_path,
)


def main():
    load_cli_args()
    args = init_cli_args()
    cohort = ""
    process_and_serialize_raw_data(args, get_preprocessed_df_path(args), cohort)


if __name__ == "__main__":
    main()
