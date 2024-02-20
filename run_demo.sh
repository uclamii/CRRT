#!/bin/bash 

export TRIALS=1
export TRAIN_VAL_COHORT=ucla_crrt
export EVAL_COHORT=ucla_crrt
export RUN_NAME=ucla_to_ucla_tune${TRIALS}
export HEAD=<PATH_TO_DEMO_FOLDER>
python ./module_code/scripts/train_and_eval.py --model-type static --seed 42 --experiment static_learning --serialization parquet --train-val-cohort ${TRAIN_VAL_COHORT} --experiment-tracking True --eval-cohort ${EVAL_COHORT} --run-name ${RUN_NAME} --local-log-path ${HEAD}/mlflow/${RUN_NAME} --ucla-crrt-data-dir ${HEAD}/UCLA --ucla-control-data-dir ${HEAD}/Controls --cedars-crrt-data-dir ${HEAD}/Cedars --test-split-size 0.2 --val-split-size 0.25 --tune-n-trials ${TRIALS} --static-metrics ['auroc','ap','brier'] --static-curves ['calibration_curve','roc_curve','pr_curve','']