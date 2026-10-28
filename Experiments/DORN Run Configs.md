# DORN run configs

22 runs: 11 setups, each with a "wrong" twin. One config file. Start from `cohorts_creation/dorn_config.yaml`, set the values from the row of the run you want, save, and start it.

## What these experiments are for

The DORN study collected thermal-preference votes together with sensor and wearable data to build personal comfort models. These runs apply the cohort approaches of the CCM paper to DORN's 20 participants, with the model setup of the original code.

- **Cold start** forms the cohorts from a one-time survey only: sex, one personality trait, life satisfaction, or an automatic split of several survey scores. **Warm start** forms them from people's votes and personal models only. Neither uses the other's data.
- **Which features go where.** Personal models use only the features that change during the study. Cohort models and the general baseline use those plus the fixed ones (height, weight, sex).
- **The `_wrong` twin of every run is the ablation.** It shows whether putting people into the right cohort matters.

## What the DORN data is

20 people with 231 votes each (4,620 votes in total). Each vote is a thermal preference: warmer (264 votes), no change (2,620) or cooler (1,736). Each vote comes with clothing, heart rate, near-body temperature and room temperature and humidity. Each person also has survey scores: the five Big Five traits, life satisfaction (SWLS), and sex, height and weight. 10 people are male and 10 female.

## How to run one experiment

1. Open `cohorts_creation/dorn_config.yaml`. Everything not listed below is already set correctly (see "Settings shared by all runs").
2. Set the values from that run's row (table below, or the copy-paste lines further down). Set all seven columns, not only the ones that differ from your current file.
3. From the project folder, in PowerShell:

```
$env:PYTHONPATH = (Get-Location).Path
person_ai_venv\Scripts\python.exe cohorts_creation\cohort_modeling.py cohorts_creation\dorn_config.yaml
```

Results go to `Experiments/<name>`. A run first deletes any existing folder with the same name. A cold run takes about 1 minute and a warm run about 7 minutes.

## The 11 setups and their "wrong" twins

Each setup exists twice. The `_wrong` twin differs from the original in only two lines: the name gets `_wrong` and `worst_cluster` is `true`. In a cold run every new person is then put into the opposite cohort. In a warm run every new person is put into the cohort whose model fits them worst, which with two cohorts is the other one.

| Run name (`name`) | cluster_forced | cluster_algo | cluster_assign | coefficients | qns_features | worst_cluster |
| --- | --- | --- | --- | --- | --- | --- |
| **1 · Cold start, Sex**  Two cohorts: male and female (10 people each). |  |  |  |  |  |  |
| dorn_cold_sex | sex | kmeans | questions | \[0.5, 0, 0.5\] | \[\] | false |
| dorn_cold_sex_wrong | sex | kmeans | questions | \[0.5, 0, 0.5\] | \[\] | true |
| **2 · Cold start, Extraversion**  Two cohorts, split at the median of the Big Five trait extraversion. |  |  |  |  |  |  |
| dorn_cold_extraversion | extraversion | kmeans | questions | \[0.5, 0, 0.5\] | \[\] | false |
| dorn_cold_extraversion_wrong | extraversion | kmeans | questions | \[0.5, 0, 0.5\] | \[\] | true |
| **3 · Cold start, Agreeableness**  Two cohorts, split at the median of agreeableness. |  |  |  |  |  |  |
| dorn_cold_agreeableness | agreeableness | kmeans | questions | \[0.5, 0, 0.5\] | \[\] | false |
| dorn_cold_agreeableness_wrong | agreeableness | kmeans | questions | \[0.5, 0, 0.5\] | \[\] | true |
| **4 · Cold start, Conscientiousness**  Two cohorts, split at the median of conscientiousness. |  |  |  |  |  |  |
| dorn_cold_conscientiousness | conscientiousness | kmeans | questions | \[0.5, 0, 0.5\] | \[\] | false |
| dorn_cold_conscientiousness_wrong | conscientiousness | kmeans | questions | \[0.5, 0, 0.5\] | \[\] | true |
| **5 · Cold start, Emotional stability**  Two cohorts, split at the median of emotional stability. |  |  |  |  |  |  |
| dorn_cold_emotional_stability | emotional_stability | kmeans | questions | \[0.5, 0, 0.5\] | \[\] | false |
| dorn_cold_emotional_stability_wrong | emotional_stability | kmeans | questions | \[0.5, 0, 0.5\] | \[\] | true |
| **6 · Cold start, Openness to experiences**  Two cohorts, split at the median of openness to experiences. |  |  |  |  |  |  |
| dorn_cold_openness_to_experiences | openness_to_experiences | kmeans | questions | \[0.5, 0, 0.5\] | \[\] | false |
| dorn_cold_openness_to_experiences_wrong | openness_to_experiences | kmeans | questions | \[0.5, 0, 0.5\] | \[\] | true |
| **7 · Cold start, Life satisfaction (SWLS)**  Two cohorts, split at the median satisfaction-with-life score. All 20 people; an extremes-only variant is below. |  |  |  |  |  |  |
| dorn_cold_swls | swls | kmeans | questions | \[0.5, 0, 0.5\] | \[\] | false |
| dorn_cold_swls_wrong | swls | kmeans | questions | \[0.5, 0, 0.5\] | \[\] | true |
| **8 · Cold start, all surveys**  KMeans finds two cohorts from six survey scores: the five Big Five traits and SWLS (list A). |  |  |  |  |  |  |
| dorn_cold_survey | '-1' | kmeans | questions | \[0.5, 0, 0.5\] | list A | false |
| dorn_cold_survey_wrong | '-1' | kmeans | questions | \[0.5, 0, 0.5\] | list A | true |
| **9 · Cold start, Big Five**  KMeans finds two cohorts from the five Big Five traits only (list B). |  |  |  |  |  |  |
| dorn_cold_big5 | '-1' | kmeans | questions | \[0.5, 0, 0.5\] | list B | false |
| dorn_cold_big5_wrong | '-1' | kmeans | questions | \[0.5, 0, 0.5\] | list B | true |
| **10 · Warm start, Dist-Cross**  Cohorts come from how similar people's votes are (half the weight) and how well their personal models predict each other (half the weight). No survey data is used. This is the paper's "Dist-Cross" approach. |  |  |  |  |  |  |
| dorn_warm_Dist_Cross | '-1' | spectral | performance | \[0.5, 0, 0.5\] | \[\] | false |
| dorn_warm_Dist_Cross_wrong | '-1' | spectral | performance | \[0.5, 0, 0.5\] | \[\] | true |
| **11 · Warm start, Cross**  Cohorts come from cross-model performance only: how well each person's personal model predicts the others. This is the paper's "Cross" approach. |  |  |  |  |  |  |
| dorn_warm_Cross | '-1' | spectral | performance | \[0, 0, 1\] | \[\] | false |
| dorn_warm_Cross_wrong | '-1' | spectral | performance | \[0, 0, 1\] | \[\] | true |

differs from the original run (the \_wrong twin)set, but has no effect in that run

**qns_features lists** (the survey columns used for automatic clustering):

```
list A: [extraversion, agreeableness, conscientiousness, emotional_stability, openness_to_experiences, swls]
list B: [extraversion, agreeableness, conscientiousness, emotional_stability, openness_to_experiences]
```

The coefficients are the weights of \[vote distribution, questionnaire, cross-model performance\]. The questionnaire weight is always 0: the warm cohort formation has no questionnaire term. The Big Five columns are the five traits of the Big Five personality model: extraversion, agreeableness, conscientiousness, emotional stability and openness to experiences. `swls` is the satisfaction-with-life score.

## Copy-paste lines per setup

Valid YAML as written. In the file the coefficients are three lines (alpha, beta, gamma); the one-line form works too. For a `_wrong` twin, change the name and set `worst_cluster: true`. The two feature lines are the same for every setup.

1 · Sex (dorn_cold_sex)

```
name: dorn_cold_sex
cluster_forced: sex
cluster_algo: kmeans
cluster_assign: questions
coefficients: [0.5, 0, 0.5]
qns_features: []
worst_cluster: false
variable_features: [clothing, heartRate, t-nb, t-ubi, rh-ubi]
fixed_features: [height, weight, sex]
extremes_filter: null
```

2 · Extraversion (dorn_cold_extraversion)

```
name: dorn_cold_extraversion
cluster_forced: extraversion
cluster_algo: kmeans
cluster_assign: questions
coefficients: [0.5, 0, 0.5]
qns_features: []
worst_cluster: false
variable_features: [clothing, heartRate, t-nb, t-ubi, rh-ubi]
fixed_features: [height, weight, sex]
extremes_filter: null
```

3 · Agreeableness (dorn_cold_agreeableness)

```
name: dorn_cold_agreeableness
cluster_forced: agreeableness
cluster_algo: kmeans
cluster_assign: questions
coefficients: [0.5, 0, 0.5]
qns_features: []
worst_cluster: false
variable_features: [clothing, heartRate, t-nb, t-ubi, rh-ubi]
fixed_features: [height, weight, sex]
extremes_filter: null
```

4 · Conscientiousness (dorn_cold_conscientiousness)

```
name: dorn_cold_conscientiousness
cluster_forced: conscientiousness
cluster_algo: kmeans
cluster_assign: questions
coefficients: [0.5, 0, 0.5]
qns_features: []
worst_cluster: false
variable_features: [clothing, heartRate, t-nb, t-ubi, rh-ubi]
fixed_features: [height, weight, sex]
extremes_filter: null
```

5 · Emotional stability (dorn_cold_emotional_stability)

```
name: dorn_cold_emotional_stability
cluster_forced: emotional_stability
cluster_algo: kmeans
cluster_assign: questions
coefficients: [0.5, 0, 0.5]
qns_features: []
worst_cluster: false
variable_features: [clothing, heartRate, t-nb, t-ubi, rh-ubi]
fixed_features: [height, weight, sex]
extremes_filter: null
```

6 · Openness to experiences (dorn_cold_openness_to_experiences)

```
name: dorn_cold_openness_to_experiences
cluster_forced: openness_to_experiences
cluster_algo: kmeans
cluster_assign: questions
coefficients: [0.5, 0, 0.5]
qns_features: []
worst_cluster: false
variable_features: [clothing, heartRate, t-nb, t-ubi, rh-ubi]
fixed_features: [height, weight, sex]
extremes_filter: null
```

7 · Life satisfaction (dorn_cold_swls)

```
name: dorn_cold_swls
cluster_forced: swls
cluster_algo: kmeans
cluster_assign: questions
coefficients: [0.5, 0, 0.5]
qns_features: []
worst_cluster: false
variable_features: [clothing, heartRate, t-nb, t-ubi, rh-ubi]
fixed_features: [height, weight, sex]
extremes_filter: null
```

8 · All surveys (dorn_cold_survey)

```
name: dorn_cold_survey
cluster_forced: '-1'
cluster_algo: kmeans
cluster_assign: questions
coefficients: [0.5, 0, 0.5]
qns_features: [extraversion, agreeableness, conscientiousness, emotional_stability, openness_to_experiences, swls]
worst_cluster: false
variable_features: [clothing, heartRate, t-nb, t-ubi, rh-ubi]
fixed_features: [height, weight, sex]
extremes_filter: null
```

9 · Big Five (dorn_cold_big5)

```
name: dorn_cold_big5
cluster_forced: '-1'
cluster_algo: kmeans
cluster_assign: questions
coefficients: [0.5, 0, 0.5]
qns_features: [extraversion, agreeableness, conscientiousness, emotional_stability, openness_to_experiences]
worst_cluster: false
variable_features: [clothing, heartRate, t-nb, t-ubi, rh-ubi]
fixed_features: [height, weight, sex]
extremes_filter: null
```

10 · Warm, Dist-Cross (dorn_warm_Dist_Cross)

```
name: dorn_warm_Dist_Cross
cluster_forced: '-1'
cluster_algo: spectral
cluster_assign: performance
coefficients: [0.5, 0, 0.5]
qns_features: []
worst_cluster: false
variable_features: [clothing, heartRate, t-nb, t-ubi, rh-ubi]
fixed_features: [height, weight, sex]
extremes_filter: null
```

11 · Warm, Cross (dorn_warm_Cross)

```
name: dorn_warm_Cross
cluster_forced: '-1'
cluster_algo: spectral
cluster_assign: performance
coefficients: [0, 0, 1]
qns_features: []
worst_cluster: false
variable_features: [clothing, heartRate, t-nb, t-ubi, rh-ubi]
fixed_features: [height, weight, sex]
extremes_filter: null
```

## Optional: life satisfaction with the extremes filter

The original CCM ran its life-satisfaction cohorts only on the people with extreme scores (at or below the 25th or at or above the 75th percentile). This variant of setup 7 does the same. On DORN the cutoffs are 16.75 and 26.5. The filter keeps **10 of the 20 people**.

```
name: dorn_cold_swls_filtered             # and ..._filtered_wrong with worst_cluster: true
cluster_forced: swls
cluster_algo: kmeans
cluster_assign: questions
coefficients: [0.5, 0, 0.5]
qns_features: []
worst_cluster: false
variable_features: [clothing, heartRate, t-nb, t-ubi, rh-ubi]
fixed_features: [height, weight, sex]
extremes_filter: {column: swls, lower_q: 0.25, upper_q: 0.75}
iterations: 10                            # 10 people stay, so at most 10 iterations
```

Compare each variant with its own general-purpose baseline. The baseline is also computed on the filtered group, so absolute numbers are not comparable across the two variants. After this run, set `extremes_filter` back to `null` and `iterations` back to 20.

## Features

Two kinds of files feed every run. The sensor and vote file `DORN_env_phy_231.csv` holds the model features and the label. The survey file `DORN_qns_b5p_calc_sex_swls_fixed.csv` holds the Big Five traits, `swls` and sex, and is only used to form cold-start cohorts. The feature keys are the same in all setups.

| Feature | Group | Config key | File | What it is |
| --- | --- | --- | --- | --- |
| clothing | variable | variable_features | DORN_env_phy_231 | Self-reported clothing level (very light, light, medium, heavy), converted to numbers |
| heartRate | variable | DORN_env_phy_231 | Heart rate |  |
| t-nb | variable | DORN_env_phy_231 | Air temperature near the body (iButton) |  |
| t-ubi | variable | DORN_env_phy_231 | Room temperature (Ubibot) |  |
| rh-ubi | variable | DORN_env_phy_231 | Room relative humidity (Ubibot) |  |
| height | fixed | fixed_features | DORN_env_phy_231 | Body height (cm) |
| weight | fixed | DORN_env_phy_231 | Body weight (kg) |  |
| sex | fixed | DORN_env_phy_231 | Male or female, converted to numbers (also in the survey file) |  |
| thermal_cozie | label | target_column | DORN_env_phy_231 | The thermal-preference vote that is predicted: 9 = warmer, 10 = no change, 11 = cooler |
| extraversion, agreeableness, conscientiousness, emotional_stability, openness_to_experiences | survey | cluster_forced, qns_features | DORN_qns_b5p_calc_sex_swls_fixed | Big Five scores from the ten-item questionnaire (TIPI), 1 to 7 in steps of 0.5. They are never an input of any prediction model. |
| swls | survey | cluster_forced, qns_features | DORN_qns_b5p_calc_sex_swls_fixed | Satisfaction with life: the sum of five items, 5 to 35. Never a model input. |

**Variable** features change during the study; **fixed** features are constant per person. Which step uses which features:

| Step | Features it uses | Note |
| --- | --- | --- |
| Personal comfort models (PCM) | the 5 variable features | One model per person. The stored PCM file is used. |
| Warm cohort formation | vote distributions and the cross-model performance of the PCMs (5 variable features) | No survey data. The weights are the `coefficients`. |
| Cold cohort formation | survey columns only: `sex`, one score, or the columns of list A or B | No sensor features. Single scores are split at the median. |
| Cohort models | the 5 variable and the 3 fixed features (8 in total) | In the sex setup `sex` is left out, because it defines the cohorts (7 features). |
| General baseline | the 5 variable and the 3 fixed features (8 in total) | Also in the sex setup, so it keeps `sex`. |

## How the data files are built

Two notebooks in `preprocessing/` build the files the runs read. Everything is read from and written to `data/dorn`.

| Notebook | Reads | Writes | What it does |
| --- | --- | --- | --- |
| preprocessing_dorn | df_info_subjects.pkl, df_cozie_env.csv | DORN_env_phy_231.csv and the other sensor files, DORN_qns\_\*.csv including DORN_qns_b5p_calc_fixed.csv | Cleans the on-boarding questionnaire and the sensor and vote data. Scores the Big Five from the ten TIPI items: answers become 1 to 7, the five reverse-scored items (reserved, critical, disorganized, anxious, conventional) are inverted as 8 minus the answer, and each trait is the mean of its two items. SWLS is the sum of its five items. It keeps 231 votes per person. |
| preprocessing (last cells) | DORN_qns_b5p_calc_fixed.csv, DORN_qns_hsps_swls_phy.csv | DORN_qns_b5p_calc_sex_swls_fixed.csv | Adds `sex` and `swls` to the Big Five scores. This is the one survey file of all runs. |

## Settings shared by all runs

These already match `dorn_config.yaml`. They are listed so you can see what a run assumes.

| Setting | Value | Meaning |
| --- | --- | --- |
| leave_1_out, iterations | true, 20 | Each of the 20 people is the "new person" once; the cohorts are built from the other 19. |
| seed | 10 | Makes a run repeatable. |
| dataset | dorn/DORN_env_phy_231.csv | Sensor and vote data: 231 votes for each of the 20 people. |
| dataset_with_time, data_set_with_time_bool | None, false | DORN has no file with timestamps, so none are attached to the saved predictions. |
| dataset_survey | dorn/DORN_qns_b5p_calc_sex_swls_fixed.csv | One row per person: the five Big Five traits, sex and swls. The same file for every run; `qns_features` picks the columns. |
| variable_features | clothing, heartRate, t-nb, t-ubi, rh-ubi | Features that change during the study. Input of the personal models and of all cohort models. |
| fixed_features | height, weight, sex | Constant per person. Input of the cohort and general models only, not of the personal models. |
| categorical_features | sex, clothing | Text columns that are converted to numbers. |
| precomputed_pcm, pcm_file | true, dorn/dict_pcm_rdf_sensor_at_f1_micro_dorn | Uses the stored personal comfort models (20 people, the five variable features) instead of training new ones. |
| excluded_users | \[\] | Nobody is left out. |
| min_total_number | 0 | Nobody is dropped for having too few votes; everyone has 231. |
| extremes_filter | null | Keeps all people. The key must be present in the file. |
| test_user_data | 1 | In warm runs all of the new person's votes are used to pick the cohort. The paper uses 1 to 7 votes. |
| parameter_finetuning | false | No hyperparameter search, to save run time. |
| cluster_num | 2 | Two cohorts in the automatic runs (KMeans in setups 8 and 9, spectral clustering in the warm runs). The forced runs split by the column itself. |
| cluster_predict_criteria | cold | The chosen cohort model is applied as it is, without retraining on the new person's data. |

## Things to know

- **Forced runs need `qns_features: []`.** A non-empty list that leaves out the forced column removes it from the survey data, and the run fails right away. This holds for the sex setup and for the single-score setups.
- **The automatic survey runs (setups 8 and 9) need their list.** The survey file contains `sex` as text, which KMeans cannot use, so these runs must name their columns (list A or B).
- **There are only 20 people.** `iterations` must not exceed the number of people kept: 20, or 10 for the filtered variant.
- **Keep `cluster_assign` as listed.** Its value ends up in the result file names (for example `dict_test_acc_questions_f1_micro`).
- **The coefficients are only used by the warm runs.** In the cold runs they are set but ignored.
- **The stored PCM file is the one to use.** `precompute_pcm.py` would overwrite it.
- **The saved PCM scores in the experiment folders are zero.** This does not affect the figures: they read the PCM from the separate `PCM/results` folder.

## Past runs

The first set of runs is saved with its configs and results in `Experiments/dorn_cold` (18 runs, the same names as above) and `Experiments/dorn_warm` (4 runs). Setup 10 was stored as `dorn_warm_Cross_Dist_survey` (plus its `_wrong` twin). The stored `dorn_warm_Dist_survey` runs match no setup here: their config has `coefficients: [1, 0, 0]`, which uses the vote distribution only and is not one of the paper's two warm approaches. Setup 11, the cross-model-only run, therefore has no run in `Experiments/dorn_warm`.

Those runs used a separate survey file for each kind of run: Big Five only, Big Five and swls, and Big Five with sex and swls. The one survey file with `qns_features` gives the same cohorts and the same scores: all 18 new cold runs match the stored ones for every person in all 20 iterations.

The stored warm runs come from code that scored every personal model on the pooled data of all people instead of on each person's own votes. The current code scores each person separately, so setup 10 gives different cohorts than the stored run in 10 of the 20 iterations (the mean cohort F1 changes from 0.676 to 0.678).

## Words used here

cold start

Cohorts are built only from survey answers. No sensor data is used to form them.

warm start

Cohorts are built from each person's own votes and personal model. No survey data is used to form them.

Dist-Cross, Cross

The two warm approaches of the paper. Dist-Cross weighs vote distribution and cross-model performance; Cross uses cross-model performance only.

cohort

A group of people. One prediction model is trained per cohort.

PCM

Personal comfort model: a model trained only on one person's data.

Big Five, TIPI

The five personality traits, measured with the ten-item questionnaire TIPI. Two items per trait; one of the two is scored in reverse.

SWLS

Satisfaction With Life Scale: five statements, each rated 1 to 7 and summed.

wrong twin

The same run with every new person assigned to the wrong cohort: in a cold run the opposite cohort, in a warm run the cohort that fits worst. It shows that the right assignment matters.

Settings follow `cohorts_creation/dorn_config.yaml` and the saved configs of the past runs in `Experiments/dorn_cold` and `Experiments/dorn_warm`, with the two warm setups named after the paper's Dist-Cross and Cross approaches. Checked on 2026-10-07 against the 22 new runs.