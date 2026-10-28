## PersonAI

# Run an experiment
Set the python path to the repo root, then start a config (PowerShell):

    $env:PYTHONPATH = "<path to the repo root>"
    python cohorts_creation\cohort_modeling.py cohorts_creation\final_config.yaml

- Configs: `final_config.yaml` (PersonAI) and `dorn_config.yaml` (DORN).
- A ready-made config for every run is in `cohorts_creation/personai_configs` and `dorn_configs`. The `make_*_configs.py` scripts generate them.
  Example: `python cohorts_creation\cohort_modeling.py cohorts_creation\dorn_configs\dorn_cold_big5.yaml`
- All runs and their settings are described in `Experiments/PersonAI Run Configs.md` and `DORN Run Configs.md`. (AI generated)
- Results go to `Experiments/<name>`. A rerun with the same name deletes the old folder first.


if one uses already existing pcm:
  - models/dict_pcm_rdf_sensor_at_f1_micro (contains pcm model, f.e. model = PersonAi )
  - (optional) dict_pcm_acc_rdf_sensor_at_cohen_kappa, dict_pcm_acc_rdf_sensor_at_f1_micro (these are in its current form not used. They contain metrics that were calculated beforehand.)

Order of execution: 
- PersonAI: 
  - -  Score calculation [Psychological Scores](https://github.com/tuw-isab/CCM_Repro_PersonAI/blob/main/preprocessing/score_calculation.py)
  - - Ghazals hr code (in [Hearate_Ghazal](https://github.com/tuw-isab/CCM_Repro_PersonAI/blob/main/preprocessing/survey_heart_rate.ipynb). Hr at survey time is always the last hr we have.)
  - - [preprocessing](https://github.com/tuw-isab/CCM_Repro_PersonAI/blob/main/preprocessing/preprocessing.ipynb)
  - - cohort_modeling.py is the experimentation script. 
  - - id014, id015, id018, id019, id041 are test users

<!-- 
- DORN: `preprocessing/preprocessing_dorn.ipynb` (reads `data/dorn/df_cozie_env.csv` and `df_info_subjects.pkl`, writes the `DORN_*` files to `data/dorn`), then `dorn_config.yaml`
-->

It executes all other scripts in order until the experiment is done and the results are saved in a new folder in Experiments/<name>. A rerun deletes an existing folder with the same name first.
Our analysis of the results are done in the plots folder

We are currently using precomputed pcms in the 'models' folder.

# Changes to original code
- Implemented leave_1_out method, instead of repeated random 80/20 splits
- finetuning is off by default with parameter_finetuning: False
- Added options for resampling
- Added timestamps to dataframe, and changed code to be able to use this
- Added saving the prediction results directly, so that metrics can be computed later on.
- Use a separate codebase for training PCMs, which can be found in PCM folder. This includes changes to the code that lead to some of the original saved scores for kappa etc to be changed to 0.    PCM accuracies in the experiment folders are placeholders and that no figure reads them
- Use leave-one-out for PMVs, with the similar configurations described in the paper. (checkout PMV folder)
- Added majority predictions
- Added logging for metadata/config
- added a condition in find_pcm that jumps a user if there is less than 5 data (analysis doesn't work... we don't use the function anyways anymore, but it is still there)
- Save Experiments in dedicated folder, not just in same folder
- In OneHotEncoder, changed the fixed one-hot structure to data-dependent one-hot structure.
- fixed Big Five reverse scoring bug.
- Prediction timestamps are taken from the same rows as the predictions
- Features are selected by name instead of dropping columns (see config options)
- run deletes an existing experiment folder with the same name. The original reused it and appended to the log.
  
# Active Branches
- main: main

# Found Bugs

- The exercising filter does nothing. The code tests met != "Exercising", but the data says exercising. It sits in the root preprocessing.ipynb (DORN sensor part) and in our preprocessing_dorn.ipynb. ([ccm code](https://github.com/buds-lab/ccm/blob/83c22fd81f824074a26183cea92895b79e8c4de7/preprocessing.ipynb)) (lowercase data read in cell 26, and the filter in cell 29)
- Big Five reverse scoring uses replace({1:7,2:6,3:5,4:4}), so answers 5 to 7 are never flipped. [ccm code](https://github.com/buds-lab/ccm/blob/83c22fd81f824074a26183cea92895b79e8c4de7/preprocessing.ipynb)(cell 14, line 38)
- The paper (section 2.7) keeps only the 25th and 75th percentile extremes for the HSPS and SWLS cohorts. The original did this with pre-filtered _ext files, not in code. This affects only the HSPS and SWLS cohorts and keeps about half the people (PersonAI 44 of 72, DORN 10 of 20). In our code this is optional through the use of the config parameter extremes_filter.

# Config options added to the original
- `variable_features` / `fixed_features`: model inputs by name. Variable ones go into all models (their order must match the stored PCMs). Fixed ones go only into the pooled models. Every other column is dropped.
- `pcm_file`: the stored PCM to load, relative to `models/`, without `.pickle`.
- `qns_features`: survey columns used for automatic clustering (`cluster_forced: '-1'`). `[]` means all columns, so one survey file can serve all runs.
- `extremes_filter`: keeps only people at or below the lower, or at or above the upper, percentile of a survey score (the paper's HSPS/SWLS rule). `null` means off.
- `category_filter`: keeps only people with certain values of a column. Optional.
- `excluded_users`: users left out of the PCM and cross-model steps (PersonAI: id016).

# Folders
- `cohorts_creation`: experiment code, configs, per-run configs
- `preprocessing`: builds the model inputs (PersonAI, DORN)
- `data`: `personai`, `dorn`
- `models`: stored PCMs
- `Experiments`: results (`PersonAI_*`, `dorn_*` and their `_old` copies) and the run descriptions
- `PCM`: the separate code and results that trained the PersonAI and DORN PCMs
- `PMV`: PMV benchmark
- `plots`, `data_exploration`: analysis and figures, exploration notebooks

## Environment
Python 3.9.13, scikit-learn 1.6.1, pandas 2.3.3, numpy 2.0.2. The stored PCMs were saved with scikit-learn 1.7.2, so loading shows a version warning. The results still reproduce.