# Project Documentation

NOTE: Everytime the preprocesing changes, you need to delete the saved preprocessed file so the whole pipeline can start again.

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