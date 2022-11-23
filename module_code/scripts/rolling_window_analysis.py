from copy import deepcopy
import sys
from os import getcwd
from os.path import join

sys.path.insert(0, join(getcwd(), "module_code"))

from utils import load_cli_args, init_cli_args
from main import main

MAX_SLIDE = 7
retrain = False

if __name__ == "__main__":
    load_cli_args()
    orig_args = init_cli_args()
    args = deepcopy(orig_args)

    # set slide_window_by 0 and run
    dargs = vars(args)
    dargs.update({"slide_window_by": 0, "max_days_on_crrt": MAX_SLIDE})
    main(args)
    # Evaluate if not tuning
    if not args.tune_n_trials:
        dargs.update(
            {"slide_window_by": 0, "stage": "eval", "max_days_on_crrt": MAX_SLIDE}
        )
        main(args)

    for i in range(1, MAX_SLIDE + 1):
        args = deepcopy(orig_args)
        dargs = vars(args)
        dargs.update({"slide_window_by": i})
        if not retrain:  # just evaluate and make sure not to tune
            dargs.update(
                {"stage": "eval", "tune_n_trials": 0, "max_days_on_crrt": MAX_SLIDE}
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
