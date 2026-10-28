
import json
from pathlib import Path
import numpy as np
import pandas
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, cohen_kappa_score, precision_score, recall_score
from matplotlib.patches import Patch

Path("../figures/statistical_evaluation").mkdir(exist_ok=True)


# Majority
#P_MAJ_ALL = "pcm_models_results/personai/majority_class_parrot_all_user_results.json"
#D_MAJ_ALL = "pcm_models_results/dorn/majority_class_parrot_all_user_results.json"

# PCM model(found in the notebook generate_confusion_matrix.ipynb, but setting both to random_forest)
P_PCM_ALL = "../../PCM/results/personai/random_forest_all_user_results.json"
P_PCM_CLASS_PARROT_ALL = "../../PCM/results/personai/majority_class_parrot_all_user_results.json"
P_PCM_DIST_PARROT_ALL = "../../PCM/results/personai/distribution_parrot_all_user_results.json"
D_PCM_ALL = "../../PCM/results/dorn/random_forest_all_user_results.json"
D_PCM_CLASS_PARROT_ALL = "../../PCM/results/dorn/majority_class_parrot_all_user_results.json"
D_PCM_DIST_PARROT_ALL = "../../PCM/results/dorn/distribution_parrot_all_user_results.json"

#Best Cohort Model Warm
P_BEST_COHORT_ALL = "../../Experiments/PersonAI_warm/PersonAI_warm_Cross_Dist_survey/meta.json"
D_BEST_COHORT_ALL = "../../Experiments/dorn_warm/dorn_warm_Dist_Cross/meta.json"
personai_best_exp = "../../Experiments/PersonAI_warm/PersonAI_warm_Cross_Dist_survey"
dorn_best_exp = "../../Experiments/dorn_warm/dorn_warm_Dist_Cross"

#Best Cohort Model Cold
P_BEST_COHORT_ALL_COLD = "../../Experiments/PersonAI_cold/PersonAI_cold_sex/meta.json"
D_BEST_COHORT_ALL_COLD = "../../Experiments/dorn_cold/dorn_cold_emotional_stability/meta.json"
personai_best_exp_cold = "../../Experiments/PersonAI_cold/PersonAI_cold_sex"
dorn_best_exp_cold = "../../Experiments/dorn_cold/dorn_cold_emotional_stability"

from statistics import median


def load_pcm_like_metrics(all_user_results_json):
    d = json.load(open(all_user_results_json, "r", encoding="utf-8"))
    users = d["all_users_results"]

    f1_micro = []
    f1_macro = []
    kappa = []
    precision_macro = []
    recall_macro = []

    for u in users:
        f1_micro.append(users[u]["average_f1_micro_over_all_folds"])
        f1_macro.append(users[u]["average_f1_macro_over_all_folds"])
        kappa.append(users[u]["average_kappa_quadratic_over_all_folds"])

        fold_precisions = []
        fold_recalls = []

        for y_true, y_pred in zip(users[u]["k_folds_y_test"], users[u]["k_folds_y_pred"]):
            fold_precisions.append(
                precision_score(y_true, y_pred, average="macro", zero_division=0)
            )
            fold_recalls.append(
                recall_score(y_true, y_pred, average="macro", zero_division=0)
            )

        precision_macro.append(np.mean(fold_precisions))
        recall_macro.append(np.mean(fold_recalls))

    return np.array(f1_micro), np.array(f1_macro), np.array(kappa), np.array(precision_macro), np.array(recall_macro)

def cohort_user_metrics(dataset_name, exp_dir, baseline=False, majority=False):
    exp_dir = Path(exp_dir)

    dfs = []
    for iter_dir in exp_dir.iterdir():
        if not iter_dir.is_dir():
            continue

        if baseline:
            csvs = list(iter_dir.glob("*_baseline_predictions.csv"))
        else:
            csvs = list(iter_dir.glob("*_cluster_predictions.csv"))

        if not csvs:
            continue

        df = pd.read_csv(csvs[0])
        df.columns = df.columns.str.strip().str.replace("\ufeff", "")
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)

    f1_micro = []
    f1_macro = []
    kappa_quad = []
    precision_macro = []
    recall_macro = []

    for u, g in df_all.groupby("user"):
        if majority:
            y_true = g["true"].astype(int).to_numpy()
            y_pred = g["majority_pred"].astype(int).to_numpy()
        else:
            y_true = g["true"].astype(int).to_numpy()
            y_pred = g["pred"].astype(int).to_numpy()

        f1_micro.append(f1_score(y_true, y_pred, average="micro"))
        f1_macro.append(f1_score(y_true, y_pred, average="macro"))
        kappa_quad.append(cohen_kappa_score(y_true, y_pred, weights="quadratic"))
        precision_macro.append(precision_score(y_true, y_pred,average="macro", zero_division=0))
        recall_macro.append(recall_score(y_true, y_pred,average="macro", zero_division=0))

    return np.array(f1_micro), np.array(f1_macro), np.array(kappa_quad), np.array(precision_macro), np.array(recall_macro)


D_maj_micro, D_maj_macro, D_maj_kappa, D_maj_pre, D_maj_recall = load_pcm_like_metrics(D_PCM_CLASS_PARROT_ALL)
D_dist_micro, D_dist_macro, D_dist_kappa, D_dist_pre, D_dist_recall = load_pcm_like_metrics(D_PCM_DIST_PARROT_ALL)

D_pcm_micro, D_pcm_macro, D_pcm_kappa, D_pcm_pre, D_pcm_recall = load_pcm_like_metrics(D_PCM_ALL)
D_co_micro, D_co_macro, D_co_kappa, D_co_pre, D_co_recall = cohort_user_metrics("dorn", dorn_best_exp)
D_co_micro_b, D_co_macro_b, D_co_kappa_b, D_co_pre_b, D_co_recall_b = cohort_user_metrics("dorn", dorn_best_exp, baseline=True)
D_co_micro_m, D_co_macro_m, D_co_kappa_m, D_co_pre_m, D_co_recall_m = cohort_user_metrics("dorn", dorn_best_exp, majority=True)

D_co_micro_cold, D_co_macro_cold, D_co_kappa_cold, D_co_pre_cold, D_co_recall_cold = cohort_user_metrics("dorn", dorn_best_exp_cold)

print("DORN users (majority):", len(D_co_macro_m))
print("DORN users (pcm):", len(D_pcm_macro))
print("DORN users (cohort):", len(D_co_macro))
print("DORN users (cohort baseline):", len(D_co_macro_b))
print("DORN users (cohort cold):", len(D_co_macro_cold))

P_maj_micro, P_maj_macro, P_maj_kappa, P_maj_pre, P_maj_recall = load_pcm_like_metrics(P_PCM_CLASS_PARROT_ALL)
P_dist_micro, P_dist_macro, P_dist_kappa, P_dist_pre, P_dist_recall = load_pcm_like_metrics(P_PCM_DIST_PARROT_ALL)

P_pcm_micro, P_pcm_macro, P_pcm_kappa, P_pcm_pre, P_pcm_recall = load_pcm_like_metrics(P_PCM_ALL)
P_co_micro, P_co_macro, P_co_kappa, P_co_pre, P_co_recall = cohort_user_metrics("personai", personai_best_exp)
P_co_micro_b, P_co_macro_b, P_co_kappa_b, P_co_pre_b, P_co_recall_b = cohort_user_metrics("personai", personai_best_exp, baseline=True)
P_co_micro_m, P_co_macro_m, P_co_kappa_m, P_co_pre_m, P_co_recall_m = cohort_user_metrics("personai", personai_best_exp, majority=True)

P_co_micro_cold, P_co_macro_cold, P_co_kappa_cold, P_co_pre_cold, P_co_recall_cold = cohort_user_metrics("personai", personai_best_exp_cold)

print("PersonAI users (majority):", len(P_co_macro_m))
print("PersonAI users (pcm):", len(P_pcm_macro))
print("PersonAI users (cohort):", len(P_co_macro))
print("PersonAI users (cohort baseline):", len(P_co_macro_b))
print("PersonAI users (cohort cold):", len(P_co_macro_cold))

from scipy.stats import ttest_ind, ttest_rel
from numpy import std, mean, sqrt

labels = ["PCM micro", "PCM macro", "Cohort Warm micro", "Cohort Warm macro", "Cohort Cold micro", "Cohort Cold macro",
          "Cohort Baseline micro", "Cohort Baseline macro", "Majority micro", "Majority macro", "PCM Majority micro",
          "PCM Majority macro", "PCM Dist micro", "PCM Dist macro"]

def cohen_d(x,y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (mean(x) - mean(y)) / sqrt(((nx-1)*std(x, ddof=1) ** 2 + (ny-1)*std(y, ddof=1) ** 2) / dof)


def estimate_necessary_sample_size(dataset, data, col_a, col_b):
    df = pandas.DataFrame(np.array(data).T, columns=labels)

    diff = df[col_a] - df[col_b]
    meand = diff.mean()
    stdd = diff.std()

    # for n ≈ (z1−α/2 + z1−β)^2 / δ^2
    # where z is the z score for confidence interval and power, alpha is aimed at 0.95, beta is aimed at 0.8
    # so the corresponding z scores are 1.96 and 0.8416.
    # sigma is the mean of the difference (the thing we want to test, so difference between two model scores) and SD is the standard deviation of that difference
    # δ = mean(d) / SD(d)
    sigm = meand/stdd
    n = (1.96+0.8416)**2 / (sigm**2)
    print(f'{dataset} {col_a}, {col_b} -> estimated minimum n: {n}')
    return n


def statistical_tests(dataset, data, colx, coly):
    df = pandas.DataFrame(np.array(data).T, columns=labels)

    x,y = df[colx], df[coly]

    stat, pval = ttest_rel(x,y, alternative='greater')
    cohend = cohen_d(x,y)
    print(f'{dataset} {colx + " vs " + coly}: pval: {pval}  cohend: {cohend} mean: {df[colx].mean()}  std: {df[colx].std()}')
    est_n = estimate_necessary_sample_size(dataset, data, colx, coly)
    return {'dataset': dataset, 'colx': colx, 'coly': coly, 'pval': pval, 'cohend': cohend, 'est_n': est_n, 'colx_mean': df[colx].mean(), 'colx_std': df[colx].std()}

if __name__ == '__main__':
    stats_res = []
    for dataset in ["dorn", "personai"]:
        if dataset == "dorn":
            data = [D_pcm_micro, D_pcm_macro, D_co_micro, D_co_macro, D_co_micro_cold, D_co_macro_cold, D_co_micro_b,
                    D_co_macro_b, D_co_micro_m, D_co_macro_m,
                    D_maj_micro, D_maj_macro, D_dist_micro, D_dist_macro]
        else:
            data = [P_pcm_micro, P_pcm_macro, P_co_micro, P_co_macro, P_co_micro_cold, P_co_macro_cold, P_co_micro_b,
                    P_co_macro_b, P_co_micro_m, P_co_macro_m,
                    P_maj_micro, P_maj_macro, P_dist_micro, P_dist_macro]

        stats_res.append(statistical_tests(dataset, data, "Cohort Baseline macro", "Majority macro"))
        stats_res.append(statistical_tests(dataset, data, "Cohort Cold macro", "Cohort Baseline macro"))
        stats_res.append(statistical_tests(dataset, data, "Cohort Warm macro", "Cohort Baseline macro"))
        stats_res.append(statistical_tests(dataset, data, "PCM Dist macro", "Cohort Warm macro"))

    # for orientation purposes - completely random predictions on a balanced dataset the F1 would be 0.33
    # asdf = np.random.randint(0,3,100000)
    # asdf2 = np.random.randint(0,3,100000)
    # sklearn.metrics.f1_score(asdf, asdf2, average="macro")
    df = pandas.DataFrame(stats_res)
    df['randF1'] = 0.3325253076977179

    print(df.to_markdown())