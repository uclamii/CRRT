# Project Documentation

##  To recreate dev environment

```
conda env create -n crrt_env --file env.yml
```

## Data preprocessing

```
run module_code/scripts/cedars_construct_features_outcomes.py
```
```
run module_code/scripts/map_cedars_* # (labs, medications, procedures)
```

To align lab units:
```
run module_code/scripts/construct_lab_unit_mappings.py
```

To preprocess the windows and go straight to evaluation:
Reference: module_code/scripts/process_and_serialize_raw_data.py

## Experiment runs

To run different experiments:
options.yml
- tune-n-trials: if >0 does tuning, Nx5 -> number of models used are 5 in each trial

```
run python scripts/rolling_window_analysis.py
```
parameters to set in scripts:
- max_days_on_crrt=7, if 7 day max window in crrt
- num_days_to_slide_fwd=7, just to slide this 7 days forward
- num_days_to_slide_bwd = -3, just to slide this 3 days backward



Everything can be triggered from `main.py` and can be adjusted in `options.yml`. 
Any flags you don't understand, search the project directory for `--<flag_name>` for its definition in ArgParse which will give an explanation.
If you want to do a rolling window analysis, execute `python scripts/rolling_window_analysis.py`.
Hyperparameter tuning is via optuna, the grid desired can be found in `module_code/exp/utils.py`.
Experiment tracking is via mlflow, optuna trials are numbered starting from 0.

NOTE: Everytime the preprocessing changes, you need to delete the saved preprocessed file so the whole pipeline can start again.
The `main.py` logic will generate and serialized a preprocessed file if it can't find one.
You can also run `python scripts/process_and_serialize_raw_data.py` to manually override any serialized files (without having to delete).
Instructions for how to use that are within that script (i.e. running preproc+serialization for each slided window for `rolling_window_analysis.py`).
In `options.yml` you should delineate  a directory per cohort, i.e., `ucla-crrt-data-dir`, `ucla-control-data-dir`, and `cedars-crrt-data-dir` to directories on your local machine.
When you preprocess and serialize these cohorts, the preprocssed files will go into these directories respectively.
When loading data the script will know where to look.

## File Organization
```
module code |
Handle loading data / preprocess.
            |--- data   |               
            Main data loading logic. Loads static + longitudinal features, combines with outcomes.
                        |--- argeparse_utils.py                
                        Responsible for parsing and formatting command-line arguments for ADTs (List, Dict).
                        |--- base_loaders.py                
                        We load data differently for static vs dynamic/longitudinal models.  Commonality is abstracted out in base_loaders.
                        |--- load.py                
                        In charge of logic for loading all features (outcomes, longitudinal, static).  Includes logic for some, but not all, feature engineering.
                        |--- longitudinal_features.py
                        Loading/Processing logic for longitudinal features such as diagnoses, problems, procedures, etc.
                        |--- longitudinal_utils.py
                        Helpful functions for time windows, aggregating repeated measurements and converting codes to be less granular.
                        |--- preproces.py
                        Basic, on-the-fly general data preprocessing before running experiments.
                        |--- sklearn_loaders.py
                        Logic to load data for sklearn-style pipelines (i.e. static models).
                        |--- torch_loaders.py
                        Logic to load data for pytorch-style pipelines (i.e. dynamic/longitudinal models).
                        |--- utils.py
                        Data loading utils in general (reading files, basic "homegrown" data transforms, etc.)
            |- evaluate |
            Logic for model evaluation (error analysis, feature importance, etc.)
            |--- exp    |
            Logic for experiments.
                        |--- utils.py
                        Help running experiments, such as hyperparameter grid definition, and experiment running (e.g. optuna).
            |--- models |
            Models used for predictive task
                        |--- base_model.py
                        Abstraction for static/dynamic models and a base wrapper class for sklearn style .fit/transform().
                        |--- longitudinal_models.py
                        Longitudinal pytorch models and its sklearn wrapper class.
                        |--- static_models.py
                        Static models across sklearn, xgboost, and lightgbm and its sklearn wrapper class.
            |--- scripts |
            One-off scripts to run before, or after experiments.
                        |--- deidentify_and_construct_features_outcomes.py
                        Deidentify Davita data for CRRT outcomes, slightly sanitize the data, and construct Age and Start Date. Age is constructed from DOB from Patient Identifiers, which is more reliable than "AGE" from Patient Demographics.
                        |--- process_and_serialize_raw_data.py
                        For static data we need to aggregate X days at a time, or maybe we change the data loading process for data range we already have, we can manually run the preprocessing steps here.
            |--- tests |
            |--- main.py
            Runs everything: loads the data and runs the predictive task with cv.  Produces a log file.
            |--- rolling_window_analysis.sh
            Runs the pipeline but with sliding the windows down.
            |--- utils.py
            CLI arg utils to help run the script and incorporate any options/YAML files/etc.
notebooks   |
All notebooks here. Exploratory, or to generate figures, etc.
```

## Data
Data should be handled with care in a secure environment.
All the relevant files are on the UCLA Health Box folder.

## Version Control
We are currently using UCLA Health Azure environment, in the `CRRT` repo.


## Production notes:

### Data preprocessing

Modify the preprocess and serialize script to run on the current day and the outcomes column used by the pipeline

### Model Testing

Load model as in static learning.py, evaluating on the testing data and return model predictions.