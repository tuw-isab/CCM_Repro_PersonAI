"""
One-time precompute of each participant's personal comfort model (PCM).

A user's PCM is trained only on that user's own rows, so it does not depend
on which other user is held out as the test participant for a given
leave-one-out iteration. This script trains every user's PCM once, on their
full data, and saves the result to the three pickle files that
`cross_model_performance` (user_profile.py) loads when `precomputed_pcm: True`.
Run it once before a leave-one-out experiment to avoid retraining every
user's model in every iteration.

Usage:
    py cohorts_creation\\precompute_pcm.py cohorts_creation\\<config>.yaml
"""
import json
import os
import pickle
import sys
import warnings

import yaml

sys.path.append(os.path.abspath(".."))
from anchor import data_dir, src_dir

from data import DataLoader
from user_profile import UserProfile


def warn(*args, **kwargs):
    pass


warnings.warn = warn

with open(str(sys.argv[1]), "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

dataset_name = config["dataset_name"]
dataset_str = config["dataset"]
dataset_qns_str = config["dataset_survey"]
dataset_type = config["dataset_type"]
dataset_with_time = config["dataset_with_time"]
data_set_with_time_bool = config["data_set_with_time_bool"]

seed = config["seed"]
target_column = config["target_column"]
target_values = config["target_values"]
leave_1_out = config["leave_1_out"]
train_test_ratio = config["train_test_ratio"]
categorical_features = config["categorical_features"]
feedback_sim = config["feedback_sim"]
qns_encoding_idx = config["qns_encoding_idx"]
qns_categories = config["qns_categories"]
gamma = config["gamma"]

divergence = config["divergence"]
qns_sim = config["qns_sim"]

model = config["model"]
scorer = config["scorer"]
use_val = config["use_val"]
resample_ccm = config["resample_ccm"]
SMOTENC_class = config["SMOTENC_class"]
SMOTENC_user = config["SMOTENC_user"]
SMOTENC_all_class = config["SMOTENC_all_class"]
parameter_finetuning = config["parameter_finetuning"]

class_limit_remove = config["class_limit_remove"]
class_limit_no_resampling = config["class_limit_no_resampling"]
class_min_resampling = config["class_min_resampling"]
downsample = config["downsample"]
downsample_60 = config["downsample_60"]
min_total_number = config["min_total_number"]

# features: the PCM only uses the variable ones, but DataLoader still needs the
# full feature_columns list (variable + fixed) to reduce the encoded dataframe
variable_features = config["variable_features"]
fixed_features = config["fixed_features"]
qns_features = config["qns_features"]
excluded_users = config["excluded_users"]
# stem of the PCM pickle, relative to models/ and without the .pickle extension -
# the single source of truth for this path, matching cohort_modeling.py
pcm_file = config["pcm_file"]

DATA_PATH = data_dir()
BASE_PATH = src_dir()

dataset_path = os.path.join(DATA_PATH, dataset_with_time if data_set_with_time_bool else dataset_str)
dataset_survey_path = os.path.join(DATA_PATH, dataset_qns_str)

# DataLoader needs a valid experiment folder (leave-one-out reads/writes its meta.json),
# even though the resulting train/test split itself is unused here.
experiment_path = os.path.join(BASE_PATH, "Experiments", f"pcm_precompute_{dataset_name}")
os.makedirs(experiment_path, exist_ok=True)
with open(os.path.join(experiment_path, "meta.json"), "w") as f:
    json.dump({"test_ids": []}, f)

print(f"Loading {dataset_path} ...")
data_loader = DataLoader(
    dataset_path,
    dataset_survey_path,
    experiment_path,
    target_column,
    target_values,
    categorical_features,
    qns_encoding_idx,
    qns_categories,
    gamma,
    leave_1_out,
    train_test_ratio,
    [],
    seed,
    SMOTENC_class=SMOTENC_class,
    SMOTENC_user=SMOTENC_user,
    class_limit_remove=class_limit_remove,
    class_limit_no_resampling=class_limit_no_resampling,
    class_min_resampling=class_min_resampling,
    downsample=downsample,
    downsample_60=downsample_60,
    min_total_number=min_total_number,
    SMOTENC_all_class=SMOTENC_all_class,
    feature_columns=variable_features + fixed_features,
    qns_features=qns_features,
    extremes_filter=None,  # PCMs are per person, so always precomputed for all occupants
    category_filter=None,
)

user_profile = UserProfile(
    feedback_sim,
    divergence,
    qns_sim,
    model,
    scorer,
    use_val,
    experiment_path,
    False,  # precomputed_pcm: irrelevant here, find_pcm is called directly
    data_loader.train_users,
    dataset_type,
    dataset_name,
    resample_ccm,
    parameter_finetuning,
    variable_features=variable_features,
    pcm_file=pcm_file,  # unused by find_pcm, but required by the constructor
    excluded_users=excluded_users,
)

print(f"Training PCM for {data_loader.df_full_encoded['user_id'].nunique()} users ...")
dict_pcm, dict_pcm_acc = user_profile.find_pcm(
    data_loader.df_full_encoded, resample_ccm=resample_ccm, parameter_finetuning=parameter_finetuning
)
print(f"Trained {len(dict_pcm)} PCMs (users with < 5 rows are skipped, see prints above)")

models_dir = os.path.join(BASE_PATH, "models")
# pcm_file may include a subfolder (e.g. 'dorn/dict_pcm_...'); make sure it exists
os.makedirs(os.path.dirname(os.path.join(models_dir, pcm_file)), exist_ok=True)


def save(name, obj):
    path = os.path.join(models_dir, name + ".pickle")
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print("Saved:", path)


save(pcm_file, dict_pcm)
save(f"{pcm_file}_acc_f1_micro", dict_pcm_acc["f1_micro"])
save(f"{pcm_file}_acc_cohen_kappa", dict_pcm_acc["cohen_kappa"])
