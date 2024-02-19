from copy import deepcopy
import sys
from os import getcwd
from os.path import join
import asyncio
from asyncio.subprocess import PIPE

sys.path.insert(0, join(getcwd(), "module_code"))

from data.longitudinal_utils import get_delta
from cli_utils import load_cli_args, init_cli_args
from main import main


if __name__ == "__main__":
    load_cli_args()
    orig_args = init_cli_args()
    args = deepcopy(orig_args)

    # user defined parameter
    max_days_on_crrt = 7
    # set slide_window_by 0 and run
    dargs = vars(args)
    dargs.update(
        {
            "rolling_evaluation": True,
            "slide_window_by": 0,
            "max_days_on_crrt": max_days_on_crrt,
            "plot_names": [],  # ["shap_explain", "randomness", "error_viz"],
        }
    )
    main(args)

    # Evaluate if not tuning
    if not args.tune_n_trials:
        dargs.update(
            {
                "rolling_evaluation": True,
                "slide_window_by": 0,
                "stage": "eval",
                "max_days_on_crrt": max_days_on_crrt,
            }
        )
        main(args)
