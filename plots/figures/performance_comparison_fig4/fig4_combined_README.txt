This figure compares the performance of different experiments using the F1-micro score. 
Both datasets results are combined in one plot.

Dataset: DORN and PersonAI

Experiments:
    - PCM: random_forest
    - General Purpose: baseline predictions l1o experiments
    - PMV: Predicted Mean Vote (leave-one-out approach)
    - Majority Class: majority_pred from the cluster predictions of l1o experiments
    - Wrong Cluster experiments: true cluster predictions of cold experiments for each feature together with their incorrect cluster assignments (~feature)
    - warm start all features: true cluster predictions of warm experiments containing all features(Person_AI_warm and dorn_warm)

Plots:
    - The F1-micro score is computed for each iteration (or fold) of every experiment.
    - Each boxplot shows the distribution of these scores across iterations.
    - The red shaded span highlights the region between the median performance of the General Purpose model and the PCM model.
    - The cold and warm experiments are indicated with blue and orange colors.
