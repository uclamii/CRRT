"""
Wrapper around module_code/scripts/rolling_window_analysis.py
"""

import subprocess

USE_ROLLING_WINDOW = False

if __name__ == "__main__":
    train_eval_pairs = [
        ("ucla_crrt", "ucla_crrt"),
        # ("cedars_crrt", "cedars_crrt"),
        # ("ucla_crrt", "ucla_control"),
        # ("ucla_crrt", "cedars_crrt"),
        # ("cedars_crrt", "ucla_crrt"),
        # ("ucla_crrt+cedars_crrt", "ucla_crrt+cedars_crrt"),
    ]
    for train_val_cohort, eval_cohort in train_eval_pairs:
        command = [
            "python",
            (
                "module_code/scripts/rolling_window_analysis.py"
                if USE_ROLLING_WINDOW
                else "module_code/main.py"
            ),
            "--train_val_cohort",
            train_val_cohort,
            "--eval_cohort",
            eval_cohort,
        ]
        subprocess.run(command)
        # print(command)
