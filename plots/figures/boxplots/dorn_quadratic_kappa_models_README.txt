This figure shows the distribution of per-user Quadratic Weighted Cohen's Kappa scores for different models.
Quadratic Kappa measures how closely predictions match the true labels, considering the distance between classes. 
Higher values indicate better agreement.

Dataset:
    DORN

Models:
    Majority: majority prediction from the experiments(majority_pred) (predict each user's most frequent class)

    PCM (Personal Comfort Model): random_forest_classifier, evaluated with 5-fold cross-validation.

    Cohort Warm: Cohorting-based Warm start model using cluster predictions from dorn_warm/dorn_warm_Dist_Cross.

    Cohort Cold: Cohorting-based Cold start model using cluster predictions from dorn_cold/l1o_dorn_cold_emotional_stability_fixed(best median f1-micro score among all true clusters cold experiments).

    Cohort Baseline: Baseline predictions from the same cohort experiment.

Each box represents the interquartile range (IQR) across users.
The line inside the box shows the median.