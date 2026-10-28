import warnings
import os
import shutil
import sys
import logging
import yaml
import json
from time import time

# print(os.environ['PYTHONPATH'])

# from anchor import data_dir,src_dir

# print('it worked')
from utils import load_variable, save_variable, general_model, general_model_eval
from data import DataLoader
from user_profile import UserProfile
from cohort import Cohort
import datetime


import sys
import os
sys.path.append(os.path.abspath(".."))
from anchor import data_dir,src_dir

def warn(*args, **kwargs):
    pass

warnings.warn = warn

# load config file from CLI
with open(str(sys.argv[1]), "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# extract parameters from config file
# data
dataset_name = config["dataset_name"]
experiment_name = config["name"]
iterations = config["iterations"]
dataset_str = config["dataset"]
dataset_qns_str = config["dataset_survey"]
dataset_type = config["dataset_type"]
dataset_with_time = config["dataset_with_time"]
pcm_file = config["pcm_file"]

# parameters
seed = config["seed"]
target_column = config["target_column"]
target_values = config["target_values"]
leave_1_out = config["leave_1_out"]
train_test_ratio = config["train_test_ratio"]
test_user_data = config["test_user_data"]
categorical_features = config["categorical_features"]
feedback_sim = config["feedback_sim"]
qns_encoding_idx = config["qns_encoding_idx"]
qns_categories = config["qns_categories"]
gamma = config["gamma"]

# user profiling
divergence = config["divergence"]
qns_sim = config["qns_sim"]

# modeling
model = config["model"]
scorer = config["scorer"]
use_val = config["use_val"]
resample_ccm = config["resample_ccm"]
#resample_SMOTENC = config["resample_SMOTENC"]
SMOTENC_class = config["SMOTENC_class"]
SMOTENC_user = config["SMOTENC_user"]

# cohort
coef = config["coefficients"]
cluster_algo = config["cluster_algo"]
cluster_num = config["cluster_num"]
cluster_assign = config["cluster_assign"]
cluster_predict_criteria = config["cluster_predict_criteria"]
cluster_forced = config["cluster_forced"]
worst_cluster = config["worst_cluster"]
data_set_with_time_bool = config["data_set_with_time_bool"]

# debugging
precomputed_pcm = config["precomputed_pcm"]
precomputed_test_users = config["precomputed_test_users"]

#config
class_limit_remove = config["class_limit_remove"]
class_limit_no_resampling = config["class_limit_no_resampling"]
class_min_resampling = config["class_min_resampling"]
downsample = config["downsample"]
downsample_60 = config["downsample_60"]
min_total_number = config["min_total_number"]
SMOTENC_all_class = config["SMOTENC_all_class"]

parameter_finetuning = config["parameter_finetuning"]

# features: the personal comfort models only use the variable ones, the pooled
# models (general baseline, cohort models) use variable + fixed ones
variable_features = config["variable_features"]
fixed_features = config["fixed_features"]

# user_ids left out of the PCM/cross-model computations entirely (dataset-specific quirks, e.g. missing/broken data)
excluded_users = config["excluded_users"]

# columns of dataset_survey used for automatic ('-1') clustering ([] = use every column in the file)
qns_features = config["qns_features"]

# optional occupant filter, e.g. the HSPS/SWLS extremes of the original CCM (None = all occupants)
extremes_filter = config["extremes_filter"]
category_filter = config.get("category_filter")  # optional: keep only occupants with certain values of a column (None = all)

# number_measurements = config["number_measurements"]

# data paths
DATA_PATH = data_dir()
BASE_PATH = src_dir()

if data_set_with_time_bool:
    dataset_path = os.path.join(DATA_PATH,dataset_with_time)
else: 
    dataset_path = os.path.join(DATA_PATH,dataset_str)

dataset_survey_path = os.path.join(DATA_PATH,dataset_qns_str)
experiment_path = os.path.join(BASE_PATH, "Experiments",experiment_name)

# folder for experiment results
# a rerun under the same name must not silently mix with a previous run's
# leftover files (stale iterations, an ever-appending .log file, ...), so the
# folder is fully cleared first if it already exists
try:
    #os.mkdir(os.path.join(BASE_PATH, "Experiments",experiment_name))
    print(os.path.join(BASE_PATH, "Experiments", experiment_name))
    if os.path.isdir(experiment_path):
        shutil.rmtree(experiment_path)
    os.makedirs(experiment_path, exist_ok=True)
except OSError:
    pass

# logging file
logging.basicConfig(
    filename= os.path.join(experiment_path,experiment_name  + ".log"),
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(message)s",
)


# meta file
meta_data = {"timestamp":str(datetime.datetime.now()),"test_ids":[]}
        
with open(os.path.join(experiment_path,'meta.json'), 'w') as f:
    json.dump(meta_data, f)


# config file
with open(os.path.join(experiment_path,'config.yaml'), 'w') as outfile:
    yaml.dump(config, outfile, default_flow_style=False)

# run experiment 'iterations' number of times
for i in range(0, iterations):
    iter_name = os.path.join(experiment_path,f"iter{i}_")#f"{experiment_name}/iter{i}_"
    logging.info(f"########## Iteration {i} ##########")
    print(f"########## Iteration {i} ##########")
    iteration_start = time()

    # prepare data
    logging.info("Loading datasets ...")
    dataset_start = time()

    test_users = (
        load_variable(iter_name + "test_users") if precomputed_test_users else []
    )
    print(dataset_path)
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
        test_users,
        seed + i,
        #resample_SMOTENC = resample_SMOTENC
        SMOTENC_class = SMOTENC_class,
        SMOTENC_user = SMOTENC_user,
        class_limit_remove = class_limit_remove,
        class_limit_no_resampling = class_limit_no_resampling,
        class_min_resampling = class_min_resampling,
        downsample = downsample,
        downsample_60 = downsample_60,
        min_total_number = min_total_number,
        SMOTENC_all_class = SMOTENC_all_class,
        feature_columns = variable_features + fixed_features,
        qns_features = qns_features,
        extremes_filter = extremes_filter,
        category_filter = category_filter,
    )  # seed is consistent in every iteration
    logging.info(f"Test users: {data_loader.test_users}")
    logging.info(f"Features encoded: {data_loader.total_cat_features}")
    save_variable(f"{iter_name}test_users", data_loader.test_users)

    logging.info(f"Loading datastes took : {time() - dataset_start}")
    print(f"Loading datastes took : {time() - dataset_start}")

    # forced cluster or automatic clustering
    logging.info("Finding Cohorts ...")
    print("Finding Cohorts ...")
    cohorts_start = time()
    cohorts = Cohort(
        coef,
        cluster_algo,
        seed + i,
        model,
        scorer,
        use_val,
        cluster_assign,
        cluster_predict_criteria,
        test_user_data,
        iter_name,
        data_loader.ohe,
        worst_cluster,
        dataset_with_time,
        parameter_finetuning,
    )
        
        
    clustering_start = time()
    if cluster_forced == "-1":
        logging.info("Automatic clustering ...")


        if cluster_algo in ("kprototype", "kmeans"):
            logging.info(f"Finding cohorts using: {cluster_algo}")

            # elbow plot and finding number of cluster
            cohorts.find_clusters(
                data_loader.df_qns_train_cat_encoded,
                qns_encoding_idx,
                categorical_features,
            )

            # labels for a fix number of cluster (cluster_num)
            if cluster_algo == "kprototype":
                cohorts.kprototype_clustering(
                    cluster_num,
                    data_loader.df_qns_train_cat_encoded,
                    qns_encoding_idx,
                    categorical_features,
                )
            elif cluster_algo == "kmeans":
                cohorts.kmeans_clustering(
                    cluster_num,
                    data_loader.df_qns_train,
                )
            logging.info(f"Current cluster membership: {cohorts.dict_user_cluster}")
        else:
            # feedback profiling
            logging.info("Feedback profiling ...")
            logging.info(f"Feedback similarity: {feedback_sim}")
            user_profile = UserProfile(
                feedback_sim,
                divergence,
                qns_sim,
                model,
                scorer,
                use_val,
                iter_name,
                precomputed_pcm,
                data_loader.train_users,
                dataset_type,
                dataset_name,
                resample_ccm,
                parameter_finetuning,
                variable_features = variable_features,
                pcm_file = pcm_file,
                excluded_users = excluded_users,
            )
            df_feedback_div_train = user_profile.feedback_profiling(
                data_loader.df_dist_train
            )
            df_feedback_div_test = user_profile.feedback_profiling(
                data_loader.df_dist_test
            )

            # cross-model profiling
            logging.info("Cross-model profiling ...")
            logging.info(
                f"Model: {model}, Scorer: {scorer}, Validation-Dev Set: {use_val}"
            )
            cmp_start = time() 

            (
                df_cross_performance,
                dict_pcm,
                dict_pcm_acc,
            ) = user_profile.cross_model_performance(data_loader.df_full_train_encoded)

            # save the model and accuracies for each iteration (train users)
            save_variable(
                f"{iter_name}df_cross_model_performance_{model}", df_cross_performance
            )
            # save_variable(f'{iter_name}dict_pcm_{model}', dict_pcm) # NOTE: quite heavy to save
            save_variable(
                f"{iter_name}dict_pcm_acc_{model}_f1_micro", dict_pcm_acc["f1_micro"]
            )
            save_variable(
                f"{iter_name}dict_pcm_acc_{model}_cohen_kappa",
                dict_pcm_acc["cohen_kappa"],
            )

            logging.info(f"Cross-model profiling took : {time() - cmp_start}")
            #print(f"Cross-model profiling took : {time() - cmp_start}")

            # cohort creation
            logging.info("Clustering ...")
            logging.info(
                f"Coef: Feedback-{coef[0]}, Qns-{coef[1]}, CrossPerf-{coef[2]}"
            )
            cohort_creation_start = time()
            df_total_sim = cohorts.add_sim_matrices(
                df_feedback_div_train, df_cross_performance
            )
            cohorts.find_clusters(df_total_sim)
            cohorts.spectral_clustering(cluster_num, df_total_sim)  # num of cohorts

            logging.info(f"cohort_creation took : {time() - cohort_creation_start}")
            #print(f"cohort_creation took : {time() - cohort_creation_start}")


    else:
        # cluster_forced based on a feature
        logging.info(f"Finding cohorts using: {cluster_forced}")
        cohorts.find_forced_cluster(
            cluster_forced, data_loader.df_qns_train_cat_encoded
        )

    # print("CLUSTER")
    # print(vars(cohorts))

    logging.info(f"Finding Clusters took : {time() - clustering_start}")
    #print(f"Finding Clusters took : {time() - clustering_start}")

    logging.info(f"Finding Cohorts took : {time() - cohorts_start}")
    #print(f"Finding Cohorts took : {time() - cohorts_start}")

    # majority model 
    print("MAJORITY ATTEMPT")
    cohorts.majority_classifying(data_loader.df_full_test_encoded,number_measurements = 10)

    cohorts.majority_classifying(data_loader.df_full_test_encoded,number_measurements = 5)

    cohorts.majority_classifying(data_loader.df_full_train_encoded)

    # cohort models
    logging.info("Training Cohort models ...")
    print("Training Cohort models ...")
    training_start = time()

    cohorts.cluster_models(
        data_loader.df_full_train_encoded, data_loader.df_qns_train_cat_encoded,resample_ccm = resample_ccm,parameter_finetuning=parameter_finetuning
    )

    save_variable(f"{iter_name}dict_cluster_df", cohorts.dict_cluster_df)
    save_variable(f"{iter_name}dict_cluster_qns", cohorts.dict_cluster_qns)

    # cohort assignation

    print("COHORT ASSIGNMENT!!!!")

    # print(data_loader.df_qns_test_cat_encoded)
    # print(qns_encoding_idx)
    # print(categorical_features)
    # print(data_loader.df_qns_test)
    # print(data_loader.df_full_test_encoded)

    logging.info("Cohort assignation ...")
    if cluster_forced != "-1":
        logging.info(f"Matching via {cluster_forced}")
        dict_test_label = cohorts.assign_cluster(data_loader.df_qns_test_cat_encoded)

    elif cluster_algo == "kprototype":
        logging.info(f"Matching via {cluster_algo}")
        dict_test_label = cohorts.assign_cluster(
            data_loader.df_qns_test_cat_encoded, qns_encoding_idx, categorical_features
        )

    elif cluster_assign == "questions":
        logging.info(f"Matching via {cluster_assign}")
        dict_test_label = cohorts.assign_cluster(data_loader.df_qns_test)

    elif cluster_assign in ("performance", "performance-noretrain"):
        logging.info(f"Matching via {cluster_assign}")
        dict_test_label = cohorts.assign_cluster(data_loader.df_full_test_encoded)

    save_variable(f"{iter_name}dict_test_label_{cluster_assign}", dict_test_label)

    
    logging.info(f"Training took : {time() - training_start}")
    print(f"Training took : {time() - training_start}")

    # cohort evaluation

    logging.info("Cohort evaluation ...")
    print("Cohort evaluation ...")
    ### fix timestamp position
    dict_test_acc = cohorts.cluster_predict(
        data_loader.df_full_test_encoded, dict_test_label, row_index=data_loader.test_row_index
    )


    save_variable(
        f"{iter_name}dict_test_acc_{cluster_assign}_f1_micro", dict_test_acc["f1_micro"]
    )
    save_variable(
        f"{iter_name}dict_test_acc_{cluster_assign}_cohen_kappa",
        dict_test_acc["cohen_kappa"],
    )

    # baseline holistic
    logging.info("Baseline holistic ...")
    #print("Baseline holistic ...")
    baseline_model = general_model(
        data_loader.df_full_train_encoded,
        stratified=False,
        model=model,
        scorer=scorer,
        use_val=use_val,
        folder_str=iter_name,
        resample_ccm = resample_ccm,
        parameter_finetuning = parameter_finetuning
    )

    print('general done')

    ### fix timestamp position
    dict_baseline_acc = general_model_eval(
        data_loader.df_full_test_encoded, baseline_model,iter_name, dataset_with_time, row_index=data_loader.test_row_index
    )

    # save variables
    # save_variable(f'{iter_name}baseline_model_{model}', baseline_model) # NOTE: too heavy
    save_variable(
        f"{iter_name}dict_baseline_acc_{model}_f1_micro", dict_baseline_acc["f1_micro"]
    )
    save_variable(
        f"{iter_name}dict_baseline_acc_{model}_cohen_kappa",
        dict_baseline_acc["cohen_kappa"],
    )

    # baseline pcm
    logging.info("Baseline pcm ...")
    # calculating the CV-tuned for each user is what takes most times

    if precomputed_pcm:
        
        # precomputing all PCM beforehand saves time
        logging.info("Loading precomputed pcm ...")
        dict_pcm_all = load_variable(f"models/{pcm_file}")
        dict_pcm_acc_all = {}
        try:
            dict_pcm_acc_all["f1_micro"] = load_variable(f"models/{pcm_file}_acc_f1_micro")
            dict_pcm_acc_all["cohen_kappa"] = load_variable(f"models/{pcm_file}_acc_cohen_kappa")
        except FileNotFoundError:
            # these values are currently unused (replaced by 0 below) - the file is optional
            dict_pcm_acc_all["f1_micro"], dict_pcm_acc_all["cohen_kappa"] = {}, {}

        # filter only the PCMs of test users (baseline)
        dict_baseline_pcm, dict_baseline_pcm_acc = {}, {}
        dict_baseline_pcm_acc["f1_micro"], dict_baseline_pcm_acc["cohen_kappa"] = {}, {}
        for user, _ in dict_pcm_all.items():
            if user in data_loader.test_users:
                dict_baseline_pcm[user] = dict_pcm_all[user]
                ###### CHANGED FOR READING PCM
                dict_baseline_pcm_acc["f1_micro"][user] = 0#dict_pcm_acc_all["f1_micro"][
                #    user
                #]
                dict_baseline_pcm_acc["cohen_kappa"][user] = 0#dict_pcm_acc_all[
                #    "cohen_kappa"
                #][user]
    else:
        # find pcm for each user (only the variable features are used)
        user_profile = UserProfile(
            feedback_sim,
            divergence,
            qns_sim,
            model,
            scorer,
            use_val,
            iter_name,
            precomputed_pcm,
            data_loader.train_users,
            dataset_type,
            dataset_name,
            resample_ccm,
            parameter_finetuning,
            variable_features = variable_features,
            pcm_file = pcm_file,
            excluded_users = excluded_users,
        )
        dict_baseline_pcm, dict_baseline_pcm_acc = user_profile.find_pcm(data_loader.df_full_test_encoded,resample_ccm=resample_ccm,parameter_finetuning=parameter_finetuning)


    # save model and accuracies (test users)
    # save_variable(f'{iter_name}dict_baseline_pcm_{model}', dict_baseline_pcm) # NOTE: too heavy
    save_variable(
        f"{iter_name}dict_baseline_pcm_acc_{model}_f1_micro",
        dict_baseline_pcm_acc["f1_micro"],
    )
    save_variable(
        f"{iter_name}dict_baseline_pcm_acc_{model}_cohen_kappa",
        dict_baseline_pcm_acc["cohen_kappa"],
    )

    logging.info(f"########## Finished iteration {i} ##########")
    logging.info(f"Iteration took : {time() - iteration_start}")
    print(f"Iteration took : {time() - iteration_start}")
