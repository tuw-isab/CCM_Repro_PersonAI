In this folder, all confusion matrices are plotted. The following models are used.

- PCM: random forest model, using summary for overall metrics, and all_users for user examples
- Cohort Model: predictions from the experiment *_warm_Dist_Cross(for dorn)/*_warm_Cross_Dist_survey(for personai) (pred from cluster_predictions)
- Majority Class: majority predictions from the experiment *_warm_Dist_Cross(for dorn)/*_warm_Cross_Dist_survey(for personai) (majority_pred from cluster_predictions)
- Cohort Model Baseline: baseline predictions from the experiment *_warm_Dist_Cross(for dorn)/*_warm_Cross_Dist_survey(for personai) (pred from baseline_predictions)

For both PersonAI and Dorn datasets we have a confusion matrix for each model. 
The naming is in a way that the dataset and model are both clear. 
The files ending with "_full.pdf" are the overall confusion matrices.

Additionally, there are confusion matrices plotted for example users. The process of choosing the users is as follows:
- top 3 users with the best average_f1_macro_over_all_folds for PCM (named as *_pcm_best_user_*.pdf)
- top 3 users with the most imbalanced classes for PCM (named as *_pcm_imbalanced_user_*.pdf)
- top 3 users with the most balanced classes for PCM (named as *_pcm_balanced_user_*.pdf)
- 2 random users for Cohort Model predictions (named as *_cross_dist_cohort_user_*.pdf)
- 2 random users for Baseline predictions (named as *_baseline_cohort_user_*.pdf)
- 2 random users for Majority Class (named as *_majority_class_user_*.pdf)


