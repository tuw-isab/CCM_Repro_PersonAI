This figure shows the distribution of per-user Micro and Macro F1 scores for different models.

Dataset:
    PersonAI

Models:
    Majority: majority prediction from the experiments(majority_pred) (predict each user's most frequent class)

    PCM (Personal Comfort Model): random_forest_classifier, evaluated with 5-fold cross-validation.

    Cohort Warm: Cohorting-based Warm start model using cluster predictions from PersonAI_warm/PersonAI_warm_Cross_Dist_survey.

    Cohort Cold: Cohorting-based Cold start model using cluster predictions from PersonAI_cold/PersonAI_cold_swls_score(best median f1-micro score among all true clusters cold experiments).

    Cohort Baseline: Baseline predictions from the same cohort experiment.

Legend: 
    Blue = Micro F1
    Orange = Macro F1

Each box represents the interquartile range (IQR) across users.
The line inside the box shows the median.