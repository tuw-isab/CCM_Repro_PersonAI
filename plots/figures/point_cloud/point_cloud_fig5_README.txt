In this figure, we compute a score per user, compares two model types, and plots the comparison.

Dataset: PersonAI and DORN
    - The cold experiments folders were used for agreeableness and life satisfaction(swls) results.(dorn_cold, PersonAI_cold)
    - For main results, the warm experiments folders were used.(dorn_warm, PersonAI_warm)

steps:
    - going through experiments and predictions: general purpose=baseline_predictions(base) and cohort model=cluster_predictions(coh)
    - calculating one F1-micro score of both general purpose and cohort models for each user 
    - comparing cohort vs general scores and computing the percent improvement for each user : delta = 100.0 * (coh - base) / base

The boxplot shows the distribution, and the scattered points are the individual users.
There is also another plot representing only data point with less than 200% gap between baseline and cohort predictions.
This is saved under 'point_clouds_personai_dorn_limited200.pdf'. This plot can show the points in more detail.