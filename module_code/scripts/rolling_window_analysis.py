from copy import deepcopy
import sys
from os import getcwd
from os.path import join


sys.path.insert(0, join(getcwd(), "module_code"))

from data.longitudinal_utils import get_delta
from cli_utils import load_cli_args, init_cli_args
from main import main

MAX_SLIDE = 7
retrain = False

if __name__ == "__main__":
    load_cli_args()
    orig_args = init_cli_args()
    args = deepcopy(orig_args)

    # Refer to get_optuna_grid for pre_start_delta
    max_window_tuning_size = (
        14 if args.tune_n_trials else get_delta(args.pre_start_delta).days
    )
    # set slide_window_by 0 and run
    dargs = vars(args)
    dargs.update({"slide_window_by": 0, "max_days_on_crrt": max_window_tuning_size})
    main(args)
    # Evaluate if not tuning
    if not args.tune_n_trials:
        dargs.update(
            {
                "slide_window_by": 0,
                "stage": "eval",
                "max_days_on_crrt": max_window_tuning_size,
            }
        )
        main(args)

    # this should be updated from tuning internally, or just set properly
    num_days_to_slide = get_delta(args.pre_start_delta).days
    for i in range(1, num_days_to_slide + 1):
        args = deepcopy(orig_args)
        dargs = vars(args)
        dargs.update({"slide_window_by": i})
        if not retrain:  # just evaluate and make sure not to tune
            dargs.update(
                {
                    "stage": "eval",
                    "tune_n_trials": 0,
                    "max_days_on_crrt": num_days_to_slide,
                }
            )
        main(args)


# shell script example
"""
#!/usr/bin/env bash

max_slide=1
retrain=false  # Comment to turn true
# Run normally once
python module_code/main.py --slide_window_by 0
python module_code/main.py --slide_window_by 0 --stage "eval"
for ((i=1; i <= $max_slide; i++)); do
    if $retrain; then
        python module_code/main.py --slide_window_by $i 
    fi
    python module_code/main.py --slide_window_by $i --stage "eval" --tune_n_trials 0
done
"""
