import json
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas
from sklearn import metrics
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
from personai_pcms.utils import *
from personai_pcms.constants import LABELS, FEATURE_COLS, POINT_COUNT_PER_USER, RANDOM_SEED, DATASET_PATH, \
    RESULTS_PATH, CATEGORICAL_FEATURES, MAKE_HOTD, TARGET_COL
from personai_pcms.model_getter_funcs import (
    make_logistic_regression_classifier,
    make_logistic_regression_classifier_with_standard_scaler,
    make_majority_class_parrot_classifier,
    make_random_forest_classifier,
    make_random_forest_classifier_with_standard_scaler,
    make_support_vector_classifier,
    make_ordinal_logistic_regression_classifier_with_mord_IT,
    make_ordinal_logistic_regression_classifier_with_mord_AT,
    make_ordinal_ridge_classifier_with_mord, make_distribution_parrot_classifier, make_random_classifier
)

from personai_pcms.utils.utils import NumpyAndPandasEncoder
from sklearn.preprocessing import LabelEncoder


# TODO:
#   updated heart beat features


def load_dataset():
    df = pandas.read_csv(DATASET_PATH)

    for col in CATEGORICAL_FEATURES:
        df[col] = LabelEncoder().fit_transform(df[col])
    if MAKE_HOTD:
        df['hotd'] = pandas.to_datetime(df['timestamp']).dt.hour
    return df


def k_fold_cv_user_pcm(X, y, model_getter_func):
    user_k_fold_run_details = {
        "k_folds_f1_micro": [],
        "k_folds_f1_macro": [],
        "k_folds_kappa_quadratic": [],
        "k_folds_confusion_matrix": [],
        "k_folds_y_test": [],
        "k_folds_y_pred": [],
    }
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    for train_index, test_index in kfold.split(X, y):
        X_train = X.loc[train_index]
        y_train = y.loc[train_index]
        X_test = X.loc[test_index]
        y_test = y.loc[test_index]

        # train model
        clf, _, _ = model_getter_func()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # score model
        fold_f1_micro = metrics.f1_score(y_test, y_pred, average="micro")
        fold_f1_macro = metrics.f1_score(y_test, y_pred, average="macro")
        fold_kappa_quadratic = cohen_kappa_score(
            y_test,
            y_pred,
            labels=LABELS,
            weights="quadratic",
        )
        fold_conf_matrix = confusion_matrix(y_test, y_pred, labels=LABELS)

        # collect results
        user_k_fold_run_details["k_folds_f1_micro"].append(fold_f1_micro)
        user_k_fold_run_details["k_folds_f1_macro"].append(fold_f1_macro)
        user_k_fold_run_details["k_folds_kappa_quadratic"].append(fold_kappa_quadratic)
        user_k_fold_run_details["k_folds_confusion_matrix"].append(fold_conf_matrix)
        user_k_fold_run_details["k_folds_y_test"].append(y_test.to_numpy())
        user_k_fold_run_details["k_folds_y_pred"].append(y_pred)

    # collect results totals for user_id
    user_k_fold_run_details.update({
        'average_f1_micro_over_all_folds': np.array(user_k_fold_run_details["k_folds_f1_micro"]).mean(axis=0),
        'average_f1_macro_over_all_folds': np.array(user_k_fold_run_details["k_folds_f1_macro"]).mean(axis=0),
        'average_kappa_quadratic_over_all_folds': float(np.nanmean(user_k_fold_run_details["k_folds_kappa_quadratic"])),
        'confusion_matrix_over_all_folds': np.array(user_k_fold_run_details["k_folds_confusion_matrix"]).sum(axis=0),
        'num_datapoints': len(X)
    })
    return user_k_fold_run_details


def train_and_pickle_pcm(X, y, model_getter_func, experiment_name, user_id):
    clf, _, _ = model_getter_func()
    clf.fit(X, y)
    pickle_file_path = RESULTS_PATH + f'{experiment_name}/{user_id}.pickle'
    Path(pickle_file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH + f'{experiment_name}/{user_id}.pickle', 'wb') as fd:
        pickle.dump(clf, fd)

def make_user_sample(udf):
    # create user datasets consisting of exactly POINT_COUNT_PER_USER number of points while keeping the
    # distribution/share of target classes the same as in the full users dataset
    ratio = udf[TARGET_COL].value_counts() / len(udf)
    ideal = ratio * POINT_COUNT_PER_USER
    sample_sizes = ideal.astype(int)
    remainder = POINT_COUNT_PER_USER - sample_sizes.sum()
    remainders = (ideal - sample_sizes).sort_values(ascending=False)

    for cls in remainders.index[:remainder]:
        sample_sizes[cls] += 1

    sample = pandas.concat([
        udf[udf[TARGET_COL] == cls].sample(sample_sizes[cls], replace=False, random_state=RANDOM_SEED)
        for cls in sample_sizes.index
    ]).reset_index(drop=True)
    return sample

def run_experiment_with_model(experiment_name, model_getter_func):
    df = load_dataset()
    users = sorted(list(df["user_id"].unique()))  # noqa: C414

    # run experiment for all user_ids
    all_users_results = {}
    for user in users:
        udf = df[df["user_id"] == user].copy().reset_index(drop=True)
        # skip users with less than POINT_COUNT_PER_USER points
        '''if len(udf) < POINT_COUNT_PER_USER:
            continue'''
        counts = udf[TARGET_COL].value_counts()
        if len(udf) < 5 or len(counts) < 2 or counts.iloc[1] < 2:  # every training fold needs at least 2 classes and 5 entries per user
            continue
        # udf_fixed_size = make_user_sample(udf)
        udf_fixed_size = udf
        X = udf_fixed_size[FEATURE_COLS]
        y = udf_fixed_size[TARGET_COL].astype(int)

        user_results = k_fold_cv_user_pcm(X, y, model_getter_func)
        all_users_results.update({user: user_results})
        train_and_pickle_pcm(X, y, model_getter_func, experiment_name, user)

    # calculate performance totals across all users for this experiment
    average_f1 = np.array([x["average_f1_micro_over_all_folds"] for x in all_users_results.values()]).mean()
    average_f1_macro_all_users = np.array([x["average_f1_macro_over_all_folds"] for x in all_users_results.values()]).mean()
    average_kappa_quadratic_all_users = np.array([x["average_kappa_quadratic_over_all_folds"] for x in all_users_results.values()]).mean()
    total_confusion_matrix = np.array([x["confusion_matrix_over_all_folds"] for x in all_users_results.values()]).sum(
        axis=0
    )

    # collect summary data for this experiment
    min_data_points_for_users = np.array([x["num_datapoints"] for x in all_users_results.values()]).min()
    max_data_points_for_users = np.array([x["num_datapoints"] for x in all_users_results.values()]).max()
    _, model_class, params = model_getter_func()
    model_info = {"model_class": model_class, "params": params}
    summary = {
        "average_f1_micro_over_all_folds_and_all_users": average_f1,
        "average_f1_macro_over_all_folds_and_all_users": average_f1_macro_all_users,
        "average_kappa_quadratic_over_all_folds_and_all_users": average_kappa_quadratic_all_users,
        "confusion_matrix_over_all_folds_and_all_users": total_confusion_matrix,
        "min_data_points_for_users": min_data_points_for_users,
        "max_data_points_for_users": max_data_points_for_users,
        "model_info": model_info,
        "labels": LABELS,
        "features": FEATURE_COLS,
        "datetime_of_experiment_run": datetime.now().isoformat(),
        "experiment_name": experiment_name
    }

    # store experiment results
    with open(RESULTS_PATH + experiment_name + "_summary.json", "w") as fd:
        json.dump(summary, fd, cls=NumpyAndPandasEncoder, indent=2)
    with open(RESULTS_PATH + experiment_name + "_all_user_results.json", "w") as fd:
        json.dump({"all_users_results": all_users_results, "summary": summary}, fd, cls=NumpyAndPandasEncoder, indent=2)


def run_pcm_experiment():
    run_experiment_with_model("distribution_parrot", make_distribution_parrot_classifier)
    run_experiment_with_model("random_classifier", make_random_classifier)
    run_experiment_with_model("majority_class_parrot", make_majority_class_parrot_classifier)

    run_experiment_with_model("random_forest", make_random_forest_classifier)
    run_experiment_with_model("support_vector", make_support_vector_classifier)
    run_experiment_with_model("logistic_regression", make_logistic_regression_classifier)
    run_experiment_with_model(
        "random_forest_classifier_with_standard_scaler", make_random_forest_classifier_with_standard_scaler
    )
    run_experiment_with_model(
        "logistic_regression_classifier_with_standard_scaler", make_logistic_regression_classifier_with_standard_scaler
    )
    run_experiment_with_model(
        "ordinal_logistic_regression_mord_IT", make_ordinal_logistic_regression_classifier_with_mord_IT
    )
    run_experiment_with_model("ordinal_logistic_regression_mord_AT", make_ordinal_logistic_regression_classifier_with_mord_AT)
    run_experiment_with_model("ordinal_ridge_mord", make_ordinal_ridge_classifier_with_mord)


if __name__ == "__main__":
    run_pcm_experiment()
