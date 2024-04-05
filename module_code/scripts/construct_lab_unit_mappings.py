"""
Create unit mapping between UCLA/Cedars
"""

from argparse import Namespace
import pandas as pd
import numpy as np
from os.path import join
from os import getcwd
import sys
from pickle import dump

sys.path.insert(0, join(getcwd(), "module_code"))
from data.utils import read_files_and_combine
from cli_utils import load_cli_args, init_cli_args
from data.longitudinal_features import map_labs


def construct_unit_mapping(args: Namespace):
    ucla_crrt_df = read_files_and_combine(["Labs.txt"], args.ucla_crrt_data_dir)
    ucla_units = ucla_crrt_df.groupby("COMPONENT_NAME")["REFERENCE_UNIT"]
    mode_and_count = ucla_units.agg(
        lambda x: (
            (pd.Series.mode(x)[0], w[0]) if len(w := x.value_counts()) > 0 else np.nan
        )
    ).dropna()
    unit_mapping = pd.DataFrame(
        mode_and_count.tolist(),
        columns=["mode_unit", "count"],
        index=mode_and_count.index,
    ).to_dict()
    # we have to put this in each data dir because of how the labs processing works
    for raw_data_dir in [
        args.ucla_crrt_data_dir,
        args.ucla_control_data_dir,
        args.cedars_crrt_data_dir,
    ]:
        with open(join(raw_data_dir, "unit_mappings.pkl"), "wb") as f:
            dump(unit_mapping, f)


if __name__ == "__main__":
    load_cli_args()
    args = init_cli_args()
    construct_unit_mapping(args)
