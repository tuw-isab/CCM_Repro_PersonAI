import glob
import json

import numpy as np
import pandas
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support

from personai_pcms.constants import FIGS_TABLES_PATH, RESULTS_PATH


def load_experiment_results_json(all_user_results_json_path):
    with open(all_user_results_json_path) as fd:
        all_user_results = json.load(fd)
    return all_user_results


def get_users_f1_scores(exp_json):
    all_users_results = exp_json['all_users_results']
    res_df = pandas.DataFrame([{'experiment_name': exp_json['summary']['experiment_name'], 'user_id': user_id,
                                'f1_micro': value["average_f1_micro_over_all_folds"],
                                'f1_macro': value["average_f1_macro_over_all_folds"]} for user_id, value in
                               all_users_results.items()])
    return res_df

def get_conf_matrix(exp_json):
    conf_matrix = exp_json['summary']['confusion_matrix_over_all_folds_and_all_users']
    experiment_name = exp_json['summary']['experiment_name']
    labels = exp_json['summary']['labels']
    return experiment_name, conf_matrix, labels


def compute_per_class_metrics(exp_json):
    experiment_name = exp_json["summary"]["experiment_name"]
    labels = exp_json["summary"]["labels"]
    all_users_results = exp_json["all_users_results"]

    y_true_all = []
    y_pred_all = []

    for user_res in all_users_results.items():
        user_id, data = user_res
        for y_true_fold, y_pred_fold in zip(data["k_folds_y_test"], data["k_folds_y_pred"]):
            y_true_all.extend(y_true_fold)
            y_pred_all.extend(y_pred_fold)

    if not y_true_all:
        return pandas.DataFrame()

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_all,
        y_pred_all,
        labels=labels,
        zero_division=0
    )

    return pandas.DataFrame({
        "experiment_name": experiment_name,
        "label": labels,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
    })


def evaluate_pcms():
    res = []
    conf_matrices = []
    per_class_rows = []

    for experiment_json in glob.glob(RESULTS_PATH + "*_all_user_results.json"):
        exp_json = load_experiment_results_json(experiment_json)
        print(experiment_json)
        res.append(get_users_f1_scores(exp_json))
        conf_matrices.append(get_conf_matrix(exp_json))
        pcm_df = compute_per_class_metrics(exp_json)
        if not pcm_df.empty:
            per_class_rows.append(pcm_df)
    df = pandas.concat(res).reset_index(drop=True)

    boxplot_all_experiments_f1_user_scores(df, 'f1_micro')
    quantiles_table(df, 'f1_micro')
    user_id_table(df, 'f1_micro')

    boxplot_all_experiments_f1_user_scores(df, 'f1_macro')
    quantiles_table(df, 'f1_macro')
    user_id_table(df, 'f1_macro')

    confusion_matrix_plots(conf_matrices)
    save_per_class_metrics(per_class_rows)


def quantiles_table(df, metric):
    qs = df.groupby('experiment_name').agg(
        q05=pandas.NamedAgg(metric, lambda x: x.quantile(0.05)),
        q25=pandas.NamedAgg(metric, lambda x: x.quantile(0.25)),
        q50=pandas.NamedAgg(metric, lambda x: x.quantile(0.5)),
        q75=pandas.NamedAgg(metric, lambda x: x.quantile(0.75)),
        q95=pandas.NamedAgg(metric, lambda x: x.quantile(0.95)),
        mean=pandas.NamedAgg(metric, 'mean'),
        std=pandas.NamedAgg(metric, 'std')
    )
    qs.to_csv(FIGS_TABLES_PATH + f"quantiles_table_{metric}_of_users_k_fold_all_experiments.csv")
    qs.to_latex(FIGS_TABLES_PATH + f"quantiles_table_{metric}_of_users_k_fold_all_experiments.tex",
                float_format="%.2f")


def user_id_table(df, metric):
    uidf = df.pivot(index='user_id',
                    columns='experiment_name',
                    values=metric)
    uidf.to_csv(FIGS_TABLES_PATH + f"user_id_table_{metric}_of_k_fold_all_experiments.csv")
    uidf.to_latex(FIGS_TABLES_PATH + f"user_id_table_{metric}_of_k_fold_all_experiments.tex",
                  float_format="%.2f")


def boxplot_all_experiments_f1_user_scores(df, metric):
    df.boxplot(column=metric, by='experiment_name', rot=90, figsize=(12, 12))
    plt.tight_layout()
    plt.savefig(FIGS_TABLES_PATH + f"boxplot_{metric}_of_users_k_fold_all_experiments.pdf")
    plt.show()


def confusion_matrix_plots(conf_matrices):
    fig, axes = plt.subplots(len(conf_matrices), 1, figsize=(8, 24))
    for (experiment_name, conf_matrix, labels), ax in zip(conf_matrices, axes):
        disp = ConfusionMatrixDisplay(display_labels=labels, confusion_matrix=np.array(conf_matrix))
        disp.plot(ax=ax)
        ax.set_title(experiment_name)
    plt.tight_layout()
    plt.savefig(FIGS_TABLES_PATH + "confusion_matrices_of_users_k_fold_all_experiments.pdf")
    plt.show()

def save_per_class_metrics(per_class_rows):
    if not per_class_rows:
        return

    per_class_all = pandas.concat(per_class_rows).reset_index(drop=True)
    per_class_all.to_csv(FIGS_TABLES_PATH + "per_class_metrics.csv", index=False)
    per_class_all.to_latex(FIGS_TABLES_PATH + "per_class_metrics.tex", index=False, float_format="%.2f")


if __name__ == '__main__':
    evaluate_pcms()
