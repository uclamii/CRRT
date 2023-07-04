from copy import deepcopy
import sys
from os import getcwd
from os.path import join


sys.path.insert(0, join(getcwd(), "module_code"))

from data.longitudinal_utils import get_delta
from cli_utils import load_cli_args, init_cli_args
from main import main

retrain = False

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

    # TODO: assert that args contain best eval args when evaluating

    # this should be updated from tuning internally, or just set properly
    # PP Note: when tuning is set 1, pre_Start delta is set to the best model value
    # this means the algorithm does not run for 7 days in the future. it just runs for one or non if range(1,1)
    # decided to hardcode it to 7
    num_days_to_slide_fwd = 7
    num_days_to_slide_bwd = -3
    # don't include the last day becaues potentially people with exactly N days of data will not have that much data / not be many
    # Patients with fewer days won't even appear anymore after sliding so far.
    # slide after and slide before
    for range in [range(1, num_days_to_slide_fwd), range(num_days_to_slide_bwd, 0)]:
        for i in range:
            args = deepcopy(orig_args)  # original args overwrite optimal ones
            dargs = vars(args)
            dargs.update({"slide_window_by": i})
            if not retrain:  # just evaluate and make sure not to tune
                dargs.update(
                    {
                        "stage": "eval",
                        "rolling_evaluation": True,
                        "tune_n_trials": 0,
                        "max_days_on_crrt": max_days_on_crrt,
                    }
                )
            main(args)


# shell script example
"""
#!/usr/bin/env bash

max_slide=1
retrain=false  # Comment to turn true
# Run normally once
python module_code/main.py --slide_window_by 0 --rolling_evaluation
python module_code/main.py --slide_window_by 0 --rolling_evaluation --stage "eval" --tune_n_trials 0
for ((i=1; i <= $max_slide; i++)); do
    if $retrain; then
        python module_code/main.py --slide_window_by $i --rolling_evaluation
    fi
    python module_code/main.py --slide_window_by $i --rolling_evaluation --stage "eval" --tune_n_trials 0
done
"""
