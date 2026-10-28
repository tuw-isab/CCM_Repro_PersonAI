This figure shows the distribution of per-user Macro Precision scores for different models on the DORN dataset.
Macro Precision is computed per user and averages precision equally across classes.
It measures how many predicted labels are correct, treating all classes equally.

Dataset:
    PersonAI

Models:
    Majority: majority prediction from the experiments(majority_pred) (predict each user's most frequent class)

    PCM (Personal Comfort Model): random_forest_classifier, evaluated with 5-fold cross-validation.

    Cohort Warm: Cohorting-based Warm start model using cluster predictions from PersonAI_warm/PersonAI_warm_Cross_Dist_survey.

    Cohort Cold: Cohorting-based Cold start model using cluster predictions from PersonAI_cold/PersonAI_cold_swls_score(best median f1-micro score among all true clusters cold experiments).

    Cohort Baseline: Baseline predictions from the same cohort experiment.

Each box represents the interquartile range (IQR) across users.
The line inside the box shows the median.