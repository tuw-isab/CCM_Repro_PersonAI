# PersonAI run configs

14 experiments, one config file. Start from `cohorts_creation/final_config.yaml`, set the values from the row of the run you want, save, and start it.

## How to run one experiment

1. Open `cohorts_creation/final_config.yaml`. Everything not listed below is already set correctly (see "Settings shared by all runs").
2. Set the values from that run's row (table below, or the copy-paste lines further down). Set all seven columns, not only the ones that differ from your current file.
3. From the project folder, in PowerShell:

```
$env:PYTHONPATH = (Get-Location).Path
person_ai_venv\Scripts\python.exe cohorts_creation\cohort_modeling.py cohorts_creation\final_config.yaml
```

Results go to `Experiments/<name>`. A run first deletes any existing folder with the same name. A cold-start run (72 iterations) takes about 6 to 7 minutes.

## The 7 experiments and their "wrong" twins

Each experiment exists twice. The `_wrong` twin is the ablation: every new person is deliberately put into the wrong cohort. It differs from the original in only two lines: the name gets `_wrong` and `worst_cluster` is `true`.

| Run name (`name`) | cluster_forced | cluster_algo | cluster_assign | coefficients | qns_features | worst_cluster |
| --- | --- | --- | --- | --- | --- | --- |
| **1 · Cold start, Sex**  Two cohorts: male and female (from the survey). |  |  |  |  |  |  |
| PersonAI_cold_sex | sex | kmeans | questions | \[0.5, 0, 0.5\] | \[\] | false |
| PersonAI_cold_sex_wrong | sex | kmeans | questions | \[0.5, 0, 0.5\] | \[\] | true |
| **2 · Cold start, Agreeableness**  Two cohorts, split at the median of the Big Five trait agreeableness. |  |  |  |  |  |  |
| PersonAI_cold_bfi2s_agreeableness | bfi2s_agreeableness | kmeans | questions | \[0.5, 0, 0.5\] | \[\] | false |
| PersonAI_cold_bfi2s_agreeableness_wrong | bfi2s_agreeableness | kmeans | questions | \[0.5, 0, 0.5\] | \[\] | true |
| **3 · Cold start, Life satisfaction (SWLS)**  Two cohorts, split at the median satisfaction-with-life score. |  |  |  |  |  |  |
| PersonAI_cold_swls_score | swls_score | kmeans | questions | \[0.5, 0, 0.5\] | \[\] | false |
| PersonAI_cold_swls_score_wrong | swls_score | kmeans | questions | \[0.5, 0, 0.5\] | \[\] | true |
| **4 · Cold start, all surveys**  KMeans finds two cohorts from six survey scores: SWLS and the five Big Five traits (list A). |  |  |  |  |  |  |
| PersonAI_cold_survey | '-1' | kmeans | questions | \[0.5, 0, 0.5\] | list A | false |
| PersonAI_cold_survey_wrong | '-1' | kmeans | questions | \[0.5, 0, 0.5\] | list A | true |
| **5 · Cold start, Big Five**  KMeans finds two cohorts from the five Big Five traits only (list B). |  |  |  |  |  |  |
| PersonAI_cold_big5 | '-1' | kmeans | questions | \[0.5, 0, 0.5\] | list B | false |
| PersonAI_cold_big5_wrong | '-1' | kmeans | questions | \[0.5, 0, 0.5\] | list B | true |
| **6 · Warm start, vote distribution + cross-model performance**  Cohorts come from how similar people's votes are and how well their personal models predict each other. No survey data is used. |  |  |  |  |  |  |
| PersonAI_warm_Cross_Dist_survey | '-1' | spectral | performance | \[0.5, 0, 0.5\] | list A | false |
| PersonAI_warm_Cross_Dist_survey_wrong | '-1' | spectral | performance | \[0.5, 0, 0.5\] | list A | true |
| **7 · Warm start, cross-model performance only**  Same as 6, but only the cross-model part is used (this is the paper's "Cross" approach). |  |  |  |  |  |  |
| PersonAI_warm_Dist_survey | '-1' | spectral | performance | \[0, 0, 1\] | list A | false |
| PersonAI_warm_Dist_survey_wrong | '-1' | spectral | performance | \[0, 0, 1\] | list A | true |

differs from the original run (the \_wrong twin)set, but has no effect in that run

**qns_features lists** (the survey columns used for automatic clustering):

```
list A: [swls_score, bfi2s_extraversion, bfi2s_agreeableness, bfi2s_conscientiousness, bfi2s_emotionality, bfi2s_openness]
list B: [bfi2s_extraversion, bfi2s_agreeableness, bfi2s_conscientiousness, bfi2s_emotionality, bfi2s_openness]
```

**Run 7 is named "Dist" but is the cross-model-only setup.** Its saved config has `coefficients: [0, 0, 1]`, which switches the vote-distribution part off. The table shows it as it was run. A pure vote-distribution run would be `[1, 0, 0]`; that one has not been run.

## Copy-paste lines per experiment

Valid YAML as written. In the file the coefficients are three lines (alpha, beta, gamma); the one-line form works too. For a `_wrong` twin, change the name and set `worst_cluster: true`.

1 · Sex (PersonAI_cold_sex)

```
name: PersonAI_cold_sex
cluster_forced: sex
cluster_algo: kmeans
cluster_assign: questions
coefficients: [0.5, 0, 0.5]
qns_features: []
worst_cluster: false
extremes_filter: null
```

2 · Agreeableness (PersonAI_cold_bfi2s_agreeableness)

```
name: PersonAI_cold_bfi2s_agreeableness
cluster_forced: bfi2s_agreeableness
cluster_algo: kmeans
cluster_assign: questions
coefficients: [0.5, 0, 0.5]
qns_features: []
worst_cluster: false
extremes_filter: null
```

3 · Life satisfaction (PersonAI_cold_swls_score)

```
name: PersonAI_cold_swls_score
cluster_forced: swls_score
cluster_algo: kmeans
cluster_assign: questions
coefficients: [0.5, 0, 0.5]
qns_features: []
worst_cluster: false
extremes_filter: null
```

4 · All surveys (PersonAI_cold_survey)

```
name: PersonAI_cold_survey
cluster_forced: '-1'
cluster_algo: kmeans
cluster_assign: questions
coefficients: [0.5, 0, 0.5]
qns_features: [swls_score, bfi2s_extraversion, bfi2s_agreeableness, bfi2s_conscientiousness, bfi2s_emotionality, bfi2s_openness]
worst_cluster: false
extremes_filter: null
```

5 · Big Five (PersonAI_cold_big5)

```
name: PersonAI_cold_big5
cluster_forced: '-1'
cluster_algo: kmeans
cluster_assign: questions
coefficients: [0.5, 0, 0.5]
qns_features: [bfi2s_extraversion, bfi2s_agreeableness, bfi2s_conscientiousness, bfi2s_emotionality, bfi2s_openness]
worst_cluster: false
extremes_filter: null
```

6 · Warm, vote distribution + cross-model (PersonAI_warm_Cross_Dist_survey)

```
name: PersonAI_warm_Cross_Dist_survey
cluster_forced: '-1'
cluster_algo: spectral
cluster_assign: performance
coefficients: [0.5, 0, 0.5]
qns_features: [swls_score, bfi2s_extraversion, bfi2s_agreeableness, bfi2s_conscientiousness, bfi2s_emotionality, bfi2s_openness]
worst_cluster: false
extremes_filter: null
```

7 · Warm, cross-model only (PersonAI_warm_Dist_survey)

```
name: PersonAI_warm_Dist_survey
cluster_forced: '-1'
cluster_algo: spectral
cluster_assign: performance
coefficients: [0, 0, 1]
qns_features: [swls_score, bfi2s_extraversion, bfi2s_agreeableness, bfi2s_conscientiousness, bfi2s_emotionality, bfi2s_openness]
worst_cluster: false
extremes_filter: null
```

## Optional: life satisfaction with the extremes filter

The original CCM ran its life-satisfaction and highly-sensitive-person cohorts only on the people with extreme scores (at or below the 25th or at or above the 75th percentile). This option does the same for run 3. It is new and has not been part of any past run.

```
name: PersonAI_cold_swls_score_ext        # and ..._ext_wrong with worst_cluster: true
cluster_forced: swls_score
qns_features: []
extremes_filter: {column: swls_score, lower_q: 0.25, upper_q: 0.75}
iterations: 44                            # 44 people stay, so at most 44 iterations
```

Compare each variant with its own general-purpose baseline. The baseline is also computed on the filtered group, so absolute numbers are not comparable across the two variants.

## Settings shared by all runs

These already match `final_config.yaml`. They are listed so you can see what a run assumes.

| Setting | Value | Meaning |
| --- | --- | --- |
| leave_1_out, iterations | true, 72 | Each of the 72 people is the "new person" once; the cohorts are built from the other 71. |
| seed | 42 | Makes a run repeatable. |
| dataset | personai/final_main_dataset_no_missing_ALLENC_new_bpm.csv | Sensor and vote data. |
| dataset_with_time, data_set_with_time_bool | personai/...\_with_time_and_hr_age.csv, true | The file actually loaded; it adds timestamps and the heart-rate delay feature. |
| dataset_survey | personai/cold_start_data_full.csv | One row per person with the survey scores (74 people). |
| variable_features | t_ubi, rh_ubi, hr_ubi, bpm_time_diff_log, CO2, activity | Features that change during the study. Input of the personal models and all cohort models. |
| fixed_features | sex, fitness_level, height_cm, weight_kg | Constant per person. Input of the cohort and general models only, not of the personal models. |
| precomputed_pcm, pcm_file | true, PersonAi/dict_pcm_rdf_sensor_at_f1_micro_personai | Uses the stored personal comfort models instead of training new ones. |
| excluded_users | \[id016\] | Left out of the personal-model computations. |
| parameter_finetuning | false | No hyperparameter search, to save run time. |
| cluster_num | 2 | Two cohorts. |
| cluster_predict_criteria | cold | The chosen cohort model is applied as it is, without retraining on the new person's data. |
| extremes_filter | null | New key. `null` keeps all people. It must be present in the file. |

## Things to know

- **Sex runs need `qns_features: []`.** A non-empty list removes the `sex` column from the survey data, and the run fails right away.
- **Keep `cluster_assign` as listed.** Its value ends up in the result file names (for example `dict_test_acc_questions_f1_micro`), so this keeps the names the same as in the existing runs.
- **Do not run `precompute_pcm.py` for PersonAI.** It retrains the personal models and would overwrite the stored PCM file.
- **The coefficients are only used by the warm runs.** In the cold runs they are set but ignored.
- **The saved PCM scores in the experiment folders are zero.** This does not affect the figures: they read the PCM from the separate `PCM/results` folder.

## Words used here

cold start

Cohorts are built only from survey answers. No sensor data is used to form them.

warm start

Cohorts are built from each person's own votes and personal model. No survey data is used to form them.

cohort

A group of people. One prediction model is trained per cohort.

PCM

Personal comfort model: a model trained only on one person's data.

wrong twin

The same run with every new person assigned to the opposite cohort, to show that the right assignment matters.

Values taken from the saved `config.yaml` of each run in `Experiments/PersonAI_cold` and `Experiments/PersonAI_warm`, checked on 2026-10-04.