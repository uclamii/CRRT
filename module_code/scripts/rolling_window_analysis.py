"""
Main driver script for all experiments
"""

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

retrain = False


async def read_stream_and_display(stream, display) -> bytes:
    """
    Read from stream line by line until EOF, display, and capture the lines.
    Ref: https://stackoverflow.com/a/25960956/1888794
    """
    output = []
    while True:
        line = await stream.readline()
        if not line:
            break
        output.append(line)
        display(line)  # assume it doesn't block
    return b"".join(output)


async def run_command(*cmd):
    # run a command
    process = await asyncio.create_subprocess_exec(*cmd, stdout=PIPE, stderr=PIPE)

    # stream the output as it runs
    try:
        stdout, stderr = await asyncio.gather(
            read_stream_and_display(process.stdout, sys.stdout.buffer.write),
            read_stream_and_display(process.stderr, sys.stderr.buffer.write),
        )
    except Exception:
        process.kill()
        raise
    finally:
        # wait for the process to exit
        rc = await process.wait()
    return (rc, stdout, stderr)


async def async_process_data(args, total_slides):
    command = ["python", "module_code/scripts/process_and_serialize_raw_data.py"]
    dargs = vars(args)

    """
    copy all the args below, but really only care about these
    args.ucla_crrt_data_dir, args.cedars_crrt_data_dir, args.ucla_control_data_dir,
    args.time_interval,
    args.pre_start_delta,
    args.post_start_delta,
    args.time_window_end,
    args.slide_window_by,
    """

    for name, val in dargs.items():
        if val is None:  # don't include if None
            continue
        if (
            val is False
        ):  # we don't use 'store_true' so booleans should not be included. "False" will evaluate to True
            continue
        if name == "slide_window_by":  # we will be changing this one
            continue

        # add all the other arguments
        command += [f"--{name.replace('_', '-')}", str(val)]

    commands = []
    for i in total_slides:
        commands.append(
            command
            + ["--slide-window-by", f"{i}"]
            + ["--cohort", f"{args.eval_cohort}"]  # create the eval_cohort parquets
        )

    await asyncio.gather(*[run_command(*cmd) for cmd in commands])


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

    # this should be updated from tuning internally, or just set properly
    # PP Note: when tuning is set 1, pre_Start delta is set to the best model value
    # this means the algorithm does not run for 7 days in the future. it just runs for one or non if range(1,1)
    # decided to hardcode it to 7
    num_days_to_slide_fwd = 7
    num_days_to_slide_bwd = -3

    if args.tune_n_trials:
        total_slides = list(range(0, num_days_to_slide_fwd)) + list(
            range(num_days_to_slide_bwd, 0)
        )

        loop = asyncio.get_event_loop()
        loop.run_until_complete(async_process_data(args, total_slides))
        loop.close()

    # don't include the last day becaues potentially people with exactly N days of data will not have that much data / not be many
    # Patients with fewer days won't even appear anymore after sliding so far.
    # slide after and slide before
    for range_ in [range(1, num_days_to_slide_fwd), range(num_days_to_slide_bwd, 0)]:
        for i in range_:
            slide_args = deepcopy(args)  # original args overwrite optimal ones
            dargs = vars(slide_args)
            dargs.update({"slide_window_by": i})
            if not retrain:  # just evaluate and make sure not to tune
                dargs.update(
                    {
                        "stage": "eval",
                        "rolling_evaluation": True,
                        "tune_n_trials": 0,
                        "max_days_on_crrt": max_days_on_crrt,
                        "plot_names": [],
                    }
                )
            main(slide_args)

    # TODO: can refactor/functionalize the below and above
    if len(args.preselect_features) > 0 or len(args.additional_eval_cohorts) > 0:
        if len(args.preselect_features) > 0:
            new_eval_cohorts = args.preselect_features
            original_eval_cohorts = args.eval_cohort.split("+")
            new_eval_cohorts = list(
                set(new_eval_cohorts).difference(set(original_eval_cohorts))
            )
        else:
            new_eval_cohorts = args.additional_eval_cohorts

        for cohort in new_eval_cohorts:
            new_eval_args = deepcopy(args)
            dargs = vars(new_eval_args)
            dargs.update(
                {
                    "slide_window_by": 0,
                    "rolling_evaluation": True,
                    "reference_window": True,  # resaves reference_id for new eval cohort
                    "tune_n_trials": 0,
                    "stage": "eval",
                    "eval_cohort": cohort,
                    "new_eval_cohort": True,
                    "max_days_on_crrt": max_days_on_crrt,
                    "plot_names": [],  # ["shap_explain", "randomness", "error_viz"],
                }
            )

            total_slides = list(range(0, num_days_to_slide_fwd)) + list(
                range(num_days_to_slide_bwd, 0)
            )

            loop = asyncio.new_event_loop()
            loop.run_until_complete(async_process_data(new_eval_args, total_slides))
            loop.close()

            main(new_eval_args)

            for range_ in [
                range(1, num_days_to_slide_fwd),
                range(num_days_to_slide_bwd, 0),
            ]:
                for i in range_:
                    slide_args = deepcopy(args)  # original args overwrite optimal ones
                    dargs = vars(slide_args)
                    dargs.update({"slide_window_by": i})
                    if not retrain:  # just evaluate and make sure not to tune
                        dargs.update(
                            {
                                "rolling_evaluation": True,
                                "reference_window": False,  # load from local use reference ids
                                "tune_n_trials": 0,
                                "stage": "eval",
                                "eval_cohort": cohort,
                                "new_eval_cohort": True,
                                "max_days_on_crrt": max_days_on_crrt,
                                "plot_names": [],
                            }
                        )
                    main(slide_args)


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
