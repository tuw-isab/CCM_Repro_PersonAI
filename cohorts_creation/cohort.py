import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path #added

from utils import save_variable, train_model, clf_metrics
from random import random
from kmodes.kmodes import KModes
from kmodes.kprototypes import KPrototypes
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.base import BaseEstimator, ClassifierMixin

import pickle
import sys
import os
sys.path.append(os.path.abspath(".."))
from anchor import data_dir
DATA_PATH = Path(data_dir())



# Forced-cohort columns whose values are the cohort labels themselves: no median split, and the column is dropped from the
# cohort models because it is constant within a cohort. Every other forced column is split at its median.
LABEL_FORCED_COLUMNS = ("sex",)


class Cohort:
    def __init__(
        self,
        coef=[0.3, 0.3, 0.3],
        cluster_algo="spectral",
        seed=13,
        model="rdf",
        scorer="f1_micro",
        use_val=True,
        cluster_assign="questions",
        cluster_predict_criteria="cold",
        test_user_data=1,
        folder_str="",
        ohe_qns="",
        worst_cluster=False,  # True if want to purposely assign to wrong cluster
        dataset_with_time=None, #added to get the timestamp dataset
        parameter_finetuning = True
    ):
        """Load parameters"""
        self.coef = coef
        self.cluster_algo = cluster_algo
        self.seed = seed
        self.model = model
        self.scorer = scorer
        self.use_val = use_val
        self.cluster_assign = cluster_assign
        self.cluster_predict_criteria = cluster_predict_criteria
        self.cluster_model_algo = 0  # placeholder, will be updated later
        self.test_user_data = test_user_data
        self.folder_str = folder_str  # e.g., '20201008/iter0_'
        self.ohe = ohe_qns
        self.worst_cluster = worst_cluster
        self.dataset_with_time = dataset_with_time
        self.parameter_finetuning = parameter_finetuning

        if self.cluster_algo == "kprototype":
            self.cluster_plot_name = self.cluster_algo
        else:
            self.cluster_plot_name = (
                str(self.coef[0]) + "-" + str(self.coef[1])
            )

        self.dict_user_cluster = {}
        self.dict_cluster_model = {}
        self.dict_cluster_df = {}
        self.dict_cluster_qns = {}

        self.cluster_forced = "-1"  # -1 means automatic clustering
        self.cluster_forced_cutoff = 0  # only used with forced clustering by a score (not by a label column such as sex)

    class MajorityClassParrot(ClassifierMixin, BaseEstimator):
        def __init__(self, *, pred_class=None):
            self.majority_class = pred_class

        def fit(self, X, y):
            values, counts = np.unique(y, return_counts=True)
            self.majority_class = values[np.argmax(counts)]
            return self

        def predict(self, X):
            return np.full(shape=X.shape[0], fill_value=self.majority_class)

        def get_params(self, deep=True):
            return {'majority_class': self.majority_class}
        
    def make_majority_class_parrot_classifier(self):

        clf = self.MajorityClassParrot()

        params = clf.get_params()
        model_class = clf.__class__.__name__
        return clf, model_class, params
    
    def majority_classifying(self,df,number_measurements = False):
        clf, _, _ = self.make_majority_class_parrot_classifier()
        #print(number_measurements)
        if not number_measurements:
            #print(f"NO LIMITITED SAMPLES {number_measurements}")
            X = np.array(
                df.iloc[:, 0 : df.shape[1] - 1]
                    )  # minus 1 for the target column
            y = np.array(df.iloc[:, -1])
            number_measurements_string = ''
        else:
            #print(f"LIMITED SAMPLES {number_measurements}")
            X = np.array(
                df.iloc[:number_measurements, 0 : df.shape[1] - 1]
                    )  # minus 1 for the target column
            y = np.array(df.iloc[:number_measurements, -1])
            #print(y)
            number_measurements_string = '_limited_' + str(number_measurements)

        clf.fit(X,y)


        pickle_file_path = os.path.join(self.folder_str, f'majority_class{number_measurements_string}.pickle')
        # print(pickle_file_path)
        Path(pickle_file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(pickle_file_path, 'wb') as fd:
            pickle.dump(clf, fd)


    def add_sim_matrices(self, dataframe_feedback, dataframe_cross):
        """
        Add similarities/distances matrices with their respective coefficients.
        If the matrix is a distance matrix, an RBF kernel is applied since
        affinity matrices are needed for the subsequent clustering. In an
        affinity matrix, a value of 1 means two items are identical.

        Args:
            dataframe_feedback: Feedback distance matrix
            dataframe_cross: Cross-model performance similarity matrix
        Returns:
            df_total_sim: sum of all similarity matrices with their coefs
                multiplied
        """
        # feedback is a distance, not affinity, matrix, so it's pass through an RBF kernel
        df_feedback_sim = self.RBF_kernel(dataframe_feedback)
        # cross-model performance
        df_cross_model_sim_avg = (dataframe_cross + dataframe_cross.transpose()) / 2
    
        # alpha * feedback + beta * cross_model
        df_total_sim = (
            self.coef[0] * df_feedback_sim + self.coef[2] * df_cross_model_sim_avg
        )

        return df_total_sim

    def RBF_kernel(self, dataframe, c=0):
        """Non-linear transformation for distance matrices to similarity matrices"""
        std_users = dataframe.std(axis=0)  # standard deviation of distance vector

        i = 0
        for _, row in dataframe.iterrows():
            normalised_dataframe = np.exp(
                -((dataframe - c) ** 2) / (2.0 * std_users[i] ** 2)
            )
            i += 1
        normalised_df = pd.DataFrame(
            normalised_dataframe, index=dataframe.index, columns=dataframe.columns
        )

        return normalised_df

    def find_clusters(
        self, dataframe, qns_idx=[], cat_features=[], k_range=range(2, 11)
    ):
        """
        Run three different cluster metrics.
        Silhouette Score Index (SSI):higher the better
        Calinski Harabasz Score (CHI): higher the better
        Davies Bouldin Index (DBI): lower the better

        Args:
            dataframe: Affinity dataframe (squared matrix)
                (self.cluster_algo == spectral)
                or qns dataframe (self.cluster_algo == kprototype)
            k_range: List of number of clusters to use.
            fig_name: Matplotlib figure name.

        Returns:
            metrics: Dictionary with the metric name as `keys` and
                the cluster metric as `values`.
        """
        X = dataframe.copy()

        ssi_list, chi_list, dbi_list = [], [], []

        for k in k_range:
            if self.cluster_algo == "spectral":
                if X.isna().any().any():
                    print("⚠️ NaNs detected in similarity matrix, replacing with 0.")
                    X = X.fillna(0)
                cluster_labels = self.spectral_clustering(k, X)
            elif self.cluster_algo == "kmeans":
                cluster_labels = self.kmeans_clustering(k, X)
            elif self.cluster_algo == "kprototype":
                cluster_labels = self.kprototype_clustering(k, X, qns_idx, cat_features)

            ssi_list.append(silhouette_score(X, cluster_labels))
            chi_list.append(calinski_harabasz_score(X, cluster_labels))
            dbi_list.append(davies_bouldin_score(X, cluster_labels))

        metrics = {"SSI": ssi_list, "CHI": chi_list, "DBI": dbi_list}
        metrics_x_label = {
            "SSI": "higher better",
            "CHI": "higher better",
            "DBI": "lower better",
        }

        figure, ax = plt.subplots(1, 3, figsize=(20, 6))

        for axis, key_values in zip(ax.flatten(), metrics.items()):
            axis.plot(k_range, key_values[1])
            axis.set_xlabel(metrics_x_label[key_values[0]], fontsize=20)
            axis.set_title(key_values[0], size=20)
            axis.tick_params(length=10, direction="inout", labelsize=20)

        figure.tight_layout()
        figure.savefig(
            f"{self.folder_str}{self.cluster_plot_name}.pdf", bbox_inches="tight"
        )
        plt.close()

        scores = {}
        scores["SSI"] = ssi_list
        scores["CHI"] = chi_list
        scores["DBI"] = dbi_list

        save_variable(f"{self.folder_str}cluster_metrics", scores)

        return scores

    def spectral_clustering(self, n_clusters, dataframe, assign_labels="discretize"):
        """Run Spectral Clustering"""

        df = dataframe.copy()

        spectral = SpectralClustering(
            n_clusters=n_clusters,
            assign_labels=assign_labels,
            affinity="precomputed",
            random_state=self.seed,
        )
        labels = spectral.fit_predict(df)

        # save cluster label for each user
        for label, user in zip(labels, df.columns):  # df's columns are all user_id's
            self.dict_user_cluster[user] = label

        return labels

    def kmeans_clustering(self, n_clusters, dataframe):
        """Run KMeans Clustering"""
        print("Using KMeans")
        df = dataframe.copy()

        # perform clustering
        self.cluster_model_algo = KMeans(n_clusters=n_clusters, random_state=0).fit(df)
        labels = self.cluster_model_algo.predict(df)

        # save cluster label for each user
        for label, user in zip(
            labels, df.index.tolist()
        ):  # df's indices are all user_id's
            self.dict_user_cluster[user] = label

        return labels

    def kprototype_clustering(self, n_clusters, dataframe, qns_idx, cat_features=[]):
        """Run K-Prototype Clustering"""
        df = dataframe.copy()

        # get index of cat_features columns in the dataframe, only for 'sex'
        if "sex" in cat_features and "sex" in df.columns.values:
            cat_idx = [df.columns.get_loc("sex")]
            #             df = df.drop('sex', axis=1)
            # append the index to the other categorical (qns) variables
            total_idx = qns_idx + cat_idx
        else:
            total_idx = qns_idx  # if sex is not used, don't add cat_idx

        # initialise clustering
        if sorted(total_idx) == list(range(0, len(df.columns.values))) or len(
            total_idx
        ) == len(
            df.columns.values
        ):  # sex was dropped
            print("Using KModes")
            kprototype = KModes(
                n_clusters=n_clusters, init="Huang", random_state=self.seed
            )
        else:
            print("Using KPrototype")
            kprototype = KPrototypes(
                n_clusters=n_clusters, init="Huang", random_state=self.seed
            )

        print(
            f"Clustering using {len(df.columns.values)} features: \n{df.columns.values}"
        )
        print(
            f"The categorical features {len(df.columns.values[total_idx])}: \n{df.columns.values[total_idx]}"
        )

        # perform clustering
        self.cluster_model_algo = kprototype.fit(df, categorical=total_idx)
        labels = self.cluster_model_algo.predict(df, categorical=total_idx)

        # save cluster label for each user
        for label, user in zip(
            labels, df.index.tolist()
        ):  # df's indices are all user_id's
            self.dict_user_cluster[user] = label

        return labels

    def find_forced_cluster(self, cluster_forced, dataframe_qns):
        """
        Performs clustering based on `cluster_forced` criteria

        Args:
            cluster_forced: Column name (string) that will determine the clusters
            dataframe-qns: Training qns dataframe with original column names (not encoded)

        Returns: (class variables)
            dict.user_cluster: Dictionary where the keys are `user_id` and the value is its
                respective cluster label
        """
        self.cluster_forced = cluster_forced  # update class variable
        df = dataframe_qns.copy()

        if cluster_forced not in LABEL_FORCED_COLUMNS:
            # find the cut_off (median) of this feature and insert value in new column
            self.cluster_forced_cutoff = df[cluster_forced].median()
            print(f"{cluster_forced} cluster cutoff {self.cluster_forced_cutoff}")
            df["cluster_col"] = df[cluster_forced].apply(
                lambda x: 0 if x < self.cluster_forced_cutoff else 1
            )
        else:  # == sex
            df["cluster_col"] = df[cluster_forced]

        # update class variable of dictionary {user: cluster_label}
        for index, row in df.iterrows():
            # each row in a qns dataframe consist of a single user
            # print(
            #     f"Training data: {index} has value of {row[cluster_forced]}, so is assigned to {row['cluster_col']}"
            # )
            self.dict_user_cluster[index] = row["cluster_col"]

        print(
            f"Available cohors in the train dataset {np.unique(list(self.dict_user_cluster.values()))}"
        )

    def cluster_models(self, dataframe, dataframe_qns,resample_ccm,parameter_finetuning):
        """
        Partitions the data according to cluster memebership and trains a model
        with CV
        Args:
            dataframe: Training dataframe already encoded
            dataframe_qns: Training qns dataframe already encoded

        Returns (class variables):
            dict_cluster_model: Dictionary with keys as the cluster label and
                the CV-tuned model as values
            dict_cluster_df: Dictionary with keys as the cluster label and the
                full data + qns + label dataframe of its members
            dict_cluster_qns: Dictionary with keys as the cluster label and the
                qns dataframe of its memebers
        """
        
        clusters_labels = np.unique(list(self.dict_user_cluster.values()))

        # filter data for each cluster and train model
        for curr_cluster in clusters_labels:
            # get all users in cluster `label`
            user_list = []
            for user, label in self.dict_user_cluster.items():
                if curr_cluster == label:
                    user_list.append(user)

            # update dictionary where the key is the cluster_label and the
            # value is the dataframe
            self.dict_cluster_df[curr_cluster] = dataframe[
                dataframe["user_id"].isin(user_list)
            ]
            self.dict_cluster_qns[curr_cluster] = dataframe_qns[
                dataframe_qns.index.isin(user_list)
            ]

            # train cluster model
            df = self.dict_cluster_df[curr_cluster]
            # .drop(
            #     ["user_id"], axis=1
            # )  # holistic model, now done in train model

            # remove the column (from training dataframe) that was used for
            # cluster_forced, if any
            if self.cluster_forced in LABEL_FORCED_COLUMNS:
                df = df.drop(columns=[self.cluster_forced], errors="ignore")

            plot_name = (
                self.folder_str + "cluster_" + str(curr_cluster) + "_" + self.model
            )

            tuned_model, _, _ = train_model(
                df,
                False,  # stratified
                self.model,
                self.scorer,
                self.use_val,
                plot_name,
                resample_ccm=resample_ccm,
                parameter_finetuning=self.parameter_finetuning,
            )
            self.dict_cluster_model[curr_cluster] = tuned_model
            #print(self.dict_cluster_model)

    def assign_cluster(self, dataframe, qns_idx=[], cat_features=[]):
        """
        Assign users from `dataframe` to a cluster following the `assign`
        criteria or by using the `cluster_forced` column from the qns dataframe

        Args:
            dataframe: Dataframe of test users
            cat_idx: Indices of categorical columns in dataframe, used with
                K-prototype
        Returns:
            dict_cluster_label: Dictionary with `user_id` as `keys` and the
                assigned label as `values`
        """
        df = dataframe.copy()
        if (
            self.cluster_assign == "questions"
            or self.cluster_forced != "-1"
            or self.cluster_algo in ("kprototype", "kmeans")
        ):
            user_id_list = df.index.tolist()
        else:  # cluster_assign == 'performance'
            user_id_list = df["user_id"].unique()

        clusters_labels = np.unique(list(self.dict_user_cluster.values()))
        dict_cluster_label = {}

        # assign the label to each user in `dataframe`
        if self.cluster_forced != "-1":
            # dataframe consists of qns for test users with indices as user_id
            for user, qns in df.iterrows():
                if self.cluster_forced in LABEL_FORCED_COLUMNS:
                    all_clusters = list(clusters_labels).copy()
                    print(f"Available clusters: {all_clusters}")
                    # remove correct cluster from list
                    all_clusters.remove(qns[self.cluster_forced])
                    wrong_cluster = all_clusters[0]  # only works for 2 clusters
                    # assign cluster to user
                    dict_cluster_label[user] = (
                        wrong_cluster
                        if self.worst_cluster
                        else qns[self.cluster_forced]
                    )
                    print(f"User forced cluster is: {qns[self.cluster_forced]}")
                    print(f"Assigned cluster is: {dict_cluster_label[user]}")
                else:
                    all_clusters = list(clusters_labels).copy()
                    print(f"Available clusters: {all_clusters}")
                    # assign based on median value saved in `cluster_forced_cutoff`
                    if self.worst_cluster:
                        # purposely assign to wrong cluster
                        dict_cluster_label[user] = (
                            1
                            if qns[self.cluster_forced] < self.cluster_forced_cutoff
                            else 0
                        )
                        print(f"Wrong label: {dict_cluster_label[user]}")
                    else:
                        # correct cluster, matches the definition in `find_forced_cluster`
                        dict_cluster_label[user] = (
                            0
                            if qns[self.cluster_forced] < self.cluster_forced_cutoff
                            else 1
                        )
                        print(f"Correct label: {dict_cluster_label[user]}")
                    print(f"{self.cluster_forced} value is {qns[self.cluster_forced]}")
                    print(f"Assigned label: {dict_cluster_label[user]}")

        # no need to use the questions, cluster_model can predict directly
        elif self.cluster_algo in ("kprototype", "kmeans"):
            # get index of cat_features columns in the dataframe, only for 'sex'
            if "sex" in cat_features and "sex" in df.columns.values:
                cat_idx = [df.columns.get_loc("sex")]
                # append the index to the other categorical (qns) variables
                total_idx = qns_idx + cat_idx
            else:
                total_idx = qns_idx  # if sex is not used, don't add cat_idx

            # predict label for all the df, where each row it's a user
            if self.cluster_algo == "kprototype":
                predicted_labels = self.cluster_model_algo.predict(
                    df, categorical=total_idx
                )
            elif self.cluster_algo == "kmeans":
                predicted_labels = self.cluster_model_algo.predict(df)

            i = 0
            for user, _ in df.iterrows():
                all_clusters = list(clusters_labels).copy()
                print(f"Available clusters: {all_clusters}")

                right_cluster = predicted_labels[i]

                # remove correct cluster from list
                all_clusters.remove(right_cluster)
                wrong_cluster = all_clusters[0]  # only works for 2 clusters

                # assign cluster to user
                dict_cluster_label[user] = (
                    wrong_cluster if self.worst_cluster else right_cluster
                )
                print(f"User right cluster is: {right_cluster}")
                print(f"Assigned cluster is: {dict_cluster_label[user]}")
                i += 1
            # end user for loop

        elif (
            self.cluster_assign == "performance"
            or self.cluster_assign == "performance-noretrain"
        ):  # warm start
            # use users' data to predict their comfort in each cluster (dist-cross and cross)
            for user in user_id_list:
                df_user = df[df["user_id"] == user]

                if self.cluster_assign == "performance":
                    # only use a fraction of samples per user
                    df_user = df_user.sample(
                        frac=self.test_user_data, axis=0, random_state=self.seed
                    )
                elif self.cluster_assign == "performance-noretrain":
                    # only use some samples per user
                    df_user = df_user.sample(
                        n=self.test_user_data, axis=0, random_state=self.seed
                    )

                df_user = df_user.drop(["user_id"], axis=1)
                acc_list = []

                # test user performance in each cluster' model
                for _, model in self.dict_cluster_model.items():
                    # create feature matrix X and target vector y
                    X = np.array(
                        df_user.iloc[:, 0 : df_user.shape[1] - 1]
                    )  # minus 1 for the target column
                    y = np.array(df_user.iloc[:, -1])
                    # evaluate model
                    y_pred = model.predict(X)
                    acc, _ = clf_metrics(
                        y, y_pred, conf_matrix_print=False, scorer=self.scorer
                    )  # acc
                    acc_list.append(acc)

                # get idx of designated cluster: max performance (best cluster), min performance (worst cluster)
                cluster_idx = (
                    acc_list.index(min(acc_list))
                    if self.worst_cluster
                    else acc_list.index(max(acc_list))
                )
                dict_cluster_label[user] = list(self.dict_cluster_model.keys())[
                    cluster_idx
                ]
                print(f"Best cluster: {acc_list.index(max(acc_list))}")
                print(f"Worst cluster: {acc_list.index(min(acc_list))}")
                print(f"Assigned cluster: {cluster_idx}")

        elif self.cluster_assign == "random":
            # randomly choose one cluster label
            for user in user_id_list:
                dict_cluster_label[user] = random.sample(list(clusters_labels), 1)[0]

        return dict_cluster_label

    ### fix timestamp position
    def cluster_predict(self, dataframe, dict_test_label, row_index=None):
        """Use cluster model to predict the performance of each user in
        `dataframe`

        Returns:
        dict_acc: Dictionary with `keys` as the `user_id` and cluster model
            performance as `values`
        """
        import pandas as pd #added
        print("beginning of cluster_predict")

        df = dataframe.copy()
        user_id_list = df["user_id"].unique()
        dict_acc = {}
        dict_acc["f1_micro"] = {}
        dict_acc["cohen_kappa"] = {}
        #----------------------------------------------------------------added to load timestamps
        ### fix timestamp position
        df_with_time = None
        ### fix timestamp position
        # a config with no with-time dataset sets dataset_with_time: None, which
        # YAML parses as the string "None", not the None object - check for both
        if row_index is not None and self.dataset_with_time not in (None, "None"):
            ### fix timestamp position
            df_with_time = pd.read_csv(DATA_PATH / self.dataset_with_time)
        #----------------------------------------------------------------


        # remove the column that was used for cluster_forced, if any
        if self.cluster_forced in LABEL_FORCED_COLUMNS:
            df = df.drop(columns=[self.cluster_forced], errors="ignore")


        #---------------------------------------------------------------- get majority baseline
        pickle_file_path = os.path.join(self.folder_str, 'majority_class.pickle')
        with open(pickle_file_path, 'rb') as f:
            maj_clf = pickle.load(f)

        majority_prediction = str(maj_clf.predict(np.array([0]))[0])

        pickle_file_path = os.path.join(self.folder_str, 'majority_class_limited_10.pickle')
        with open(pickle_file_path, 'rb') as f:
            maj_clf_lim_10 = pickle.load(f)
        majority_prediction_limited_10 = str(maj_clf_lim_10.predict(np.array([0]))[0])

        pickle_file_path = os.path.join(self.folder_str, 'majority_class_limited_5.pickle')
        with open(pickle_file_path, 'rb') as f:
            maj_clf_lim_5 = pickle.load(f)
        majority_prediction_limited_5 = str(maj_clf_lim_5.predict(np.array([0]))[0])

        print("LOADING SUCCESS")

        for user in user_id_list:
            # prepare test user data
            df_user = df[df["user_id"] == user]
            df_user = df_user.drop(["user_id"], axis=1)

            #---------------------------------------------------------------extra
            # if questions-based label missing, skip user (no crash)
            if user not in dict_test_label:
                print(f"[warn] No questions-based label for {user}; skipping.")
                continue
            #----------------------------------------------------------------

            # prepare cluster model
            print(f"Predicted cluster for {user} is {dict_test_label[user]}")
            cluster_label = dict_test_label[user]  # test user cluster label
            cluster_model = self.dict_cluster_model[cluster_label]

            # prepare user data
            X = np.array(
                df_user.iloc[:, 0 : df_user.shape[1] - 1]
            )  # minus 1 for the target column
            y = np.array(df_user.iloc[:, -1])

            ### fix timestamp position
            if df_with_time is not None:
                ### fix timestamp position
                # df_user.index still holds this row's position in `dataframe`,
                # unchanged since cluster_predict was called
                ### fix timestamp position
                user_rows = row_index[df_user.index.to_numpy()]
                ### fix timestamp position
                timestamps = df_with_time.iloc[user_rows]["timestamp"].reset_index(drop=True)
                ### fix timestamp position
                assert len(timestamps) == len(y)
            ### fix timestamp position
            else:
                ### fix timestamp position
                timestamps = None

            if self.cluster_predict_criteria == "cold":  # use model to predict
                # evaluate model and save accuracy
                y_pred = cluster_model.predict(X)


                #------------------------------------------------------------added to save the predictions

                pred_dir = Path(f"{self.folder_str}cluster_predictions")
                pred_dir.mkdir(parents=True, exist_ok=True)

                ### fix timestamp position
                pred_data = {"user": [user] * len(y)}
                ### fix timestamp position
                if timestamps is not None:
                    ### fix timestamp position
                    pred_data["timestamp"] = timestamps
                ### fix timestamp position
                pred_data.update({
                    "true": y,
                    "pred": y_pred,
                    "majority_pred": majority_prediction,
                    "limited_majority_pred_10": majority_prediction_limited_10,
                    "limited_majority_pred_5": majority_prediction_limited_5
                })
                ### fix timestamp position
                df_pred = pd.DataFrame(pred_data)
                df_pred.to_csv(pred_dir / f"{user}_cluster_predictions.csv", index=False)

                #------------------------------------------------------------

            elif self.cluster_predict_criteria == "warm":  # retweak the model

                # get current cluster data and remove the column that was used
                # for cluster_forced, if any
                df_cluster = self.dict_cluster_df[cluster_label]
                df_cluster = df_cluster.drop(["user_id"], axis=1)
                if self.cluster_forced in LABEL_FORCED_COLUMNS:
                    df_cluster = df_cluster.drop(columns=[self.cluster_forced], errors="ignore")

                # transform it to matrix X and target vector y
                # minus 1 for the target column
                X_cluster = np.array(df_cluster.iloc[:, 0 : df_cluster.shape[1] - 1])
                y_cluster = np.array(df_cluster.iloc[:, -1])

                # use only a specific ratio of the test user data
                df_user_warm = df_user.sample(
                    frac=self.test_user_data, random_state=self.seed
                )
                # minus 1 for the target column
                X_warm = np.array(df_user_warm.iloc[:, 0 : df_user_warm.shape[1] - 1])
                y_warm = np.array(df_user_warm.iloc[:, -1])

                # concatenate cluster data and test user
                X_cluster_user = np.vstack((X_cluster, X_warm))
                y_cluster_user = np.hstack((y_cluster, y_warm))

                # train new model with concatenated data
                if self.model == "rdf":
                    cluster_model.fit(X_cluster_user, y_cluster_user)
                    # predict on all test user's test data
                    y_pred = cluster_model.predict(X)
                    #------------------------------------------------------------added to save the predictions
                    pred_dir = Path(f"{self.folder_str}cluster_predictions")
                    pred_dir.mkdir(parents=True, exist_ok=True)

                    ### fix timestamp position
                    pred_data = {"user": user}
                    ### fix timestamp position
                    if timestamps is not None:
                        ### fix timestamp position
                        pred_data["timestamp"] = timestamps
                    ### fix timestamp position
                    pred_data.update({
                        "true": y,
                        "pred": y_pred,
                        "majority_pred": majority_prediction,
                    })
                    ### fix timestamp position
                    df_pred = pd.DataFrame(pred_data)

                    df_pred.to_csv(pred_dir / f"{user}_cluster_predictions.csv", index=False)
                    #------------------------------------------------------------

            # save performance metrics
            acc_f1_micro, _ = clf_metrics(
                y, y_pred, conf_matrix_print=False, scorer="f1_micro"
            )
            acc_cohen_kappa, _ = clf_metrics(
                y, y_pred, conf_matrix_print=False, scorer="cohen_kappa"
            )
            dict_acc["f1_micro"][user] = acc_f1_micro
            dict_acc["cohen_kappa"][user] = acc_cohen_kappa

        return dict_acc