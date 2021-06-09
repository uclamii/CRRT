# Project Documentation

NOTE: Everytime the preprocesing changes, you need to delete the saved preprocessed file so the whole pipeline can start again.

## File Organization
```
module code |
            Handle loading data / preprocess.
            |--- data   |               
                        Main data loading logic. Loads static + longitudinal features, combines with outcomes.
                        |--- load.py                
                        In charge of logic for loading all longitudinal features.
                        |--- longitudinal_features.py
                        Helpful functions for aggregating repeated measurements and converting codes to be less granular.
                        |--- longitudinal_utils.py
                        Used for the CV logic.
                        |--- preproces.py
                        Data loading utils in general (reading files, data transforms, etc.)
                        |--- utils.py
            Logic for experiments.
            |--- exp    |
                        Logic for running CV on predictive task.
                        |--- cv.py
            When we build more complex predictive models the code can go here.
            |--- models |
                        Empty
            Runs everything: loads the data and runs the predictive task with cv.
            Produces a log file.
            |--- main.py
notebooks   |
            All notebooks here. Exploratory, or to generate figures, etc.
```

## Data
Data should be handled with care in a secure environment.
All the relevant files are on the UCLA Health AWS S3 bucket: `s3://endstage-renal`.

## Version Control
We are currently using UCLA Health Azure environment, in the `CRRT` repo.