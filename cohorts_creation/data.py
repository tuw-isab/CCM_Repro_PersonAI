import os
import random
import numpy as np
import pandas as pd
import json
import logging

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTENC

import sys
import os
sys.path.append(os.path.abspath(".."))
from anchor import src_dir  

class DataLoader:
    def __init__(
        self,
        dataframe_full_str,
        dataframe_qns_str,
        experiment_path,
        target_column="thermal_cozie",
        target_values=[9.0, 10.0, 11.0],
        categorical_features=[],
        qns_encoding_idx=[],
        qns_categories=[],
        gamma=0.0,
        leave_1_out = False,
        train_test_ratio=0.8,
        precomputed_list_test_users=[],
        seed=42,
        #resample_SMOTENC = False,
        SMOTENC_class = False,
        SMOTENC_user = False,
        class_limit_remove = 5,
        class_limit_no_resampling = True,
        class_min_resampling = 5,
        downsample = False,
        downsample_60 = False,
        min_total_number = 10,
        SMOTENC_all_class = False,
        *,
        feature_columns,
        qns_features,
        extremes_filter,
        category_filter = None,
    ):
        """
        Load all default parameters and creates test and train splits.
        Assumes the dataframes have a column named `user_id` and the target
        variable is under the `thermal_cozie` column.
        `feature_columns` lists the model input columns (variable + fixed
        features from the config); the encoded dataframes are reduced to
        `user_id`, these columns in this order, and the target as last column.
        `qns_features` restricts the survey/questionnaire dataframe to these
        columns ([] = keep all of them); this only matters for the automatic
        ('-1') clustering path, which otherwise clusters on every column the
        configured `dataset_survey` file happens to have. Applied here, right
        after loading, so `qns_encoding_idx` (a column-position list) keeps
        referring to one single, consistent column layout throughout.
        `extremes_filter` (None = off) keeps only the occupants at/below the
        `lower_q` or at/above the `upper_q` percentile of the survey column
        `column`, in both the sensor and the survey data (as the original CCM
        did for its HSPS/SWLS cohorts). The percentiles are computed over the
        occupants that have sensor data. Applied before `qns_features`, so the
        filter column may be left out of `qns_features`.
        `category_filter` (None = off) keeps only the occupants whose value in
        the survey column `column` is one of the values listed under `keep`,
        again in both the sensor and the survey data. Applied right after the
        extremes filter and likewise before `qns_features`.
        """
        self.df_full = pd.read_csv(dataframe_full_str)
        self.df_qns = pd.read_csv(dataframe_qns_str, index_col="user_id")
        self.extremes_filter = extremes_filter
        if self.extremes_filter:
            col = self.extremes_filter["column"]
            assert col in self.df_qns.columns, f"extremes_filter column {col} missing from {dataframe_qns_str}"
            in_experiment = self.df_qns.index.isin(self.df_full["user_id"].unique())
            scores = self.df_qns.loc[in_experiment, col]
            low = scores.quantile(self.extremes_filter["lower_q"])
            high = scores.quantile(self.extremes_filter["upper_q"])
            keep = scores.index[(scores <= low) | (scores >= high)]
            # no reset_index: the row labels must stay the row positions of the file (timestamp lookup)
            self.df_full = self.df_full[self.df_full["user_id"].isin(keep)]
            self.df_qns = self.df_qns.loc[self.df_qns.index.isin(keep)]
            print(f"extremes filter on {col}: cutoffs {low} / {high}, {len(keep)} of {len(scores)} occupants kept")
        self.category_filter = category_filter
        if self.category_filter:
            col = self.category_filter["column"]
            assert col in self.df_qns.columns, f"category_filter column {col} missing from {dataframe_qns_str}"
            in_experiment = self.df_qns.index.isin(self.df_full["user_id"].unique())
            values = self.df_qns.loc[in_experiment, col]
            keep = values.index[values.isin(self.category_filter["keep"])]
            # no reset_index: the row labels must stay the row positions of the file (timestamp lookup)
            self.df_full = self.df_full[self.df_full["user_id"].isin(keep)]
            self.df_qns = self.df_qns.loc[self.df_qns.index.isin(keep)]
            print(f"category filter on {col}: keeping {self.category_filter['keep']}, {len(keep)} of {len(values)} occupants kept, left out: {sorted(set(values.index) - set(keep))}")
        self.qns_features = list(qns_features)
        if self.qns_features:
            missing = [c for c in self.qns_features if c not in self.df_qns.columns]
            assert not missing, f"qns_features columns missing from {dataframe_qns_str}: {missing}"
            self.df_qns = self.df_qns[self.qns_features]
        self.experiment_path = experiment_path
        self.target_column = target_column
        self.target_values = target_values.copy()
        self.categorical_features = categorical_features.copy()
        self.qns_encoding_idx = qns_encoding_idx.copy()
        self.qns_categories = qns_categories.copy()
        self.gamma = gamma
        self.leave_1_out = leave_1_out
        self.train_test_ratio = train_test_ratio
        self.precomputed_list_test_users = precomputed_list_test_users.copy()
        self.seed = seed
        self.ohe = ""
        self.total_cat_features = categorical_features.copy()
        #self.resample_SMOTENC = resample_SMOTENC
        self.SMOTENC_class = SMOTENC_class
        self.SMOTENC_user = SMOTENC_user
        self.class_limit_remove = class_limit_remove
        self.class_limit_no_resampling = class_limit_no_resampling
        self.class_min_resampling = class_min_resampling
        self.downsample = downsample
        self.downsample_60 = downsample_60
        self.min_total_number = min_total_number
        self.SMOTENC_all_class = SMOTENC_all_class
        self.feature_columns = list(feature_columns)
        self.logger = logging.getLogger(__name__)

        self.logger.info("Test Loading datasets inside DataLoader....")
        print("Test Loading datasets inside DataLoader....")

        if self.downsample:
            self.set_seed()
            print("Initialising downsampling class...")
            num_counts = self.df_full.thermal_cozie.value_counts()
            print(num_counts)
            num_counts = num_counts.reset_index(drop = False)

            num_to_drop = num_counts.loc[0,"count"]-num_counts.loc[1,"count"]
            class_to_drop = num_counts.loc[0,"thermal_cozie"]

            to_drop = self.df_full.loc[self.df_full.thermal_cozie == class_to_drop].sample(n = num_to_drop,random_state = self.seed) # get seperate dataframe so that if we wasnt to use something with it later it is already there
            to_drop.to_csv("dropped_data.csv")
            self.df_full = self.df_full.loc[~self.df_full.index.isin(to_drop.index)]
            self.df_qns = self.df_qns.loc[~self.df_qns.index.isin(to_drop.index)]

            self.df_full.to_csv("downsampled_data.csv")

        if self.downsample_60:
            print("Initialising downsampling to 60 samples...")
            num_counts = self.df_full.user_id.value_counts()

            num_counts = num_counts.reset_index(drop = False)
            enough_samples = num_counts.loc[num_counts["count"] >= 60,"user_id"]
            self.df_full = self.df_full.loc[self.df_full.user_id.isin(enough_samples)]
            self.df_qns = self.df_qns.loc[self.df_qns.index.isin(enough_samples)]


            df = self.df_full.copy()
            sample_list = []    
            for id in df.user_id.unique():

                _df_id = df.loc[df.user_id == id,:]

                class_counts = _df_id.thermal_cozie.value_counts()
                class_counts_1 = class_counts[class_counts == 1].reset_index()

                if len(_df_id) == 60:
                    stratified_sample = _df_id
                #### stratified train_test_split does not work if a case has only one instance. 
                #### in this case remove single case. randomly (by distribution) decide if it should be included or not.
                #### and then either add it back, or not
                else:
                    if(len(class_counts_1) == 0):
                        stratified_sample, _ = train_test_split(_df_id, train_size=60, stratify=_df_id[['thermal_cozie']])
                    else:
                        _df_id_1 = _df_id.loc[_df_id.thermal_cozie.isin(class_counts_1.thermal_cozie)]
                        _df_id = _df_id.loc[~_df_id.thermal_cozie.isin(class_counts_1.thermal_cozie)]
                        r = random.uniform(0, 1)
                        if r <= 60/sum(class_counts) :
                            stratified_sample, _ = train_test_split(_df_id, train_size=59, stratify=_df_id[['thermal_cozie']])
                            stratified_sample = pd.concat([stratified_sample,_df_id_1])
                        else:
                            stratified_sample, _ = train_test_split(_df_id, train_size=60, stratify=_df_id[['thermal_cozie']])                    

                #to_drop = _df_id.sample(n = num_counts.loc[num_counts["user_id"] == id,"num_remove"],random_state = self.seed)
                sample_list.append(stratified_sample)
            self.df_full = pd.concat(sample_list)

        if self.min_total_number != 0:
            num_counts = self.df_full.user_id.value_counts()
            # print("Class distribution after downsampling")
            # print(num_counts)
            num_counts = num_counts.reset_index(drop = False)
            
            enough_users = num_counts.loc[num_counts["count"] >= self.min_total_number,"user_id"]
            self.df_full = self.df_full.loc[self.df_full.user_id.isin(enough_users)]
            self.df_qns = self.df_qns.loc[self.df_qns.index.isin(enough_users)]

            
        if self.class_limit_remove >0:
            print("Removing users with small classes")
            valid_participant_list = []
            df = self.df_full.copy()
            for id in df["user_id"].unique():

                _df_id = df.loc[df.user_id == id,:]

                num_counts = _df_id.thermal_cozie.value_counts() 
                if len(num_counts[num_counts >= self.class_limit_remove]) == 3:
                    valid_participant_list.append(id)
            
            self.df_full = self.df_full.loc[self.df_full.user_id.isin(valid_participant_list)]
            self.df_qns = self.df_qns.loc[self.df_qns.index.isin(valid_participant_list)]

        # train test splits
        if not self.precomputed_list_test_users:
            self.train_users, self.test_users = self.participant_train_test_split()
        else:
            self.test_users = self.precomputed_list_test_users
            self.train_users = self.train_precomputed()


        print("splitting data")
        self.df_full_train, self.df_full_test = self.dataframe_train_test_split(
            self.df_full, thermal_label=True
        )

        self.df_qns_train, self.df_qns_test = self.dataframe_train_test_split(
            self.df_qns, user_index=True
        )

        # making sure the label is the last column in the un-split dataframe
        df_full_y = self.df_full.pop(self.target_column)
        self.df_full.loc[:, self.target_column] = df_full_y

        # feedback distribution calculation
        self.df_dist_all = self.feedback_dist(self.df_full)
        self.df_dist_train = self.feedback_dist(self.df_full_train)
        self.df_dist_test = self.feedback_dist(self.df_full_test)

        print("feedback distance")

        # feature encoding
        self.dict_encoder = self.cat_encoder()  # only `categorical_features`
        self.dict_encoder_full = self.cat_encoder(
            full=True
        )  # all categorical and qns features
        self.df_full_encoded = self.select_features(self.fit_cat_encoder(self.df_full))
        self.df_full_train_encoded = self.select_features(self.fit_cat_encoder(self.df_full_train))
        self.df_full_test_encoded = self.select_features(self.fit_cat_encoder(self.df_full_test))

        self.df_qns_train_cat_encoded = self.fit_cat_encoder(
            self.df_qns_train, full=True
        )
        self.df_qns_test_cat_encoded = self.fit_cat_encoder(self.df_qns_test, full=True)

        if self.qns_encoding_idx is not None:
            self.ohe = self.onehot_encoder()
            self.df_qns_train_encoded = self.fit_onehot_encoder(self.df_qns_train)
            self.df_qns_test_encoded = self.fit_onehot_encoder(self.df_qns_test)
        else:  # no need to encode
            self.ohe = None
            self.df_qns_train_encoded = self.df_qns_train
            self.df_qns_test_encoded = self.df_qns_test

        # print("encoded")

        if (self.SMOTENC_user | self.SMOTENC_class | self.SMOTENC_all_class):


            self.df_full_train_encoded = self.SMOTENC_resampling(self.df_full_train_encoded,"thermal_cozie",self.SMOTENC_user,self.SMOTENC_class,self.SMOTENC_all_class,self.class_limit_no_resampling,self.class_min_resampling)

        #df_full_train_encoded.

    def select_features(self, dataframe):
        """Keep `user_id`, the configured feature columns and the target (as last column)"""
        columns = ["user_id"] + self.feature_columns + [self.target_column]
        missing = [c for c in columns if c not in dataframe.columns]
        assert not missing, f"columns from the config are missing in the dataset: {missing}"
        return dataframe[columns]

    def set_seed(self):
        """Set seed"""
        random.seed(self.seed)
        np.random.seed(self.seed)
        os.environ["PYTHONHASHSEED"] = str(self.seed)

    def participant_train_test_split(self):
        """
        Generate train and test split based on the list of participants.
        Assumes the participant id is under a column named `user_id`.
        This means that a participant's data is either in the train set OR test set
        but NOT on both sides.
        """
        self.set_seed()
        df = self.df_full.copy()
        list_participants = sorted(df["user_id"].unique())

        random.seed(self.seed)
        # random.shuffle(list_participants)
        print(list_participants)

        if self.leave_1_out:
            #logging.info("Leave 1 out method chosen...")

            print("Leave 1 out method chosen...")
            #BASE_PATH = src_dir()
            f = open(os.path.join(self.experiment_path,'meta.json'))#open('meta.json')
            meta_data = json.load(f)
            test_ids = meta_data["test_ids"]

            remaining_participants = sorted(list(set(list_participants) - set(test_ids)))

            print("Remaining Participants")
            print(remaining_participants)
            test_participants = random.sample(remaining_participants,1)
            print("test participant")
            print(test_participants)
            test_ids.append(test_participants[0])

            meta_data["test_ids"] = test_ids

            with open(os.path.join(self.experiment_path,'meta.json'),'w') as f: #open(os.path.join(self.experiment_path,'meta.json'), 'w') as f:
                json.dump(meta_data, f)

        # meta_data = {"id":df_id_list,
        #     "columns":list(dict.fromkeys(flatten(df_columns_list)))}
        
        else: 
        # randomly choose test_participants according to ratio
            print("Random data split chosen...")
            test_participants = random.sample(
                    set(list_participants),
                    int(round((1 - self.train_test_ratio) * len(list_participants))),
                )
            
        
        
        train_participants = list(set(list_participants) - set(test_participants))
        return train_participants, test_participants

    def train_precomputed(self):
        """Return train users based on existing list of test users"""
        df = self.df_full.copy()
        list_participants = df["user_id"].unique()

        return list(set(list_participants) - set(self.test_users))

    def dataframe_train_test_split(
        self, dataframe, user_index=False, thermal_label=False
    ):
        """Splits a dataframe according to lists of users"""
        self.set_seed()
        df = dataframe.copy()

        # filter the data
        df_train = (
            df[df["user_id"].isin(self.train_users)]
            if not user_index
            else df[df.index.isin(self.train_users)]
        )
        df_test = (
            df[df["user_id"].isin(self.test_users)]
            if not user_index
            else df[df.index.isin(self.test_users)]
        )

        if thermal_label:
            # move thermal comfort response to the end of the dataframe
            df_train_y = df_train.pop(self.target_column)
            df_test_y = df_test.pop(self.target_column)
            df_train.loc[:, self.target_column] = df_train_y
            df_test.loc[:, self.target_column] = df_test_y

        # shuffle
        if not user_index:  # only shuffle for non-qns dataframes
            ### fix timestamp position
            df_train = df_train.sample(frac=1, random_state=self.seed)
            ### fix timestamp position
            df_test = df_test.sample(frac=1, random_state=self.seed)
            ### fix timestamp position
            if thermal_label:
                ### fix timestamp position
                # each row's position in the original dataset file, kept in this
                # shuffled order, so timestamps can still be looked up afterwards
                ### fix timestamp position
                self.train_row_index = df_train.index.to_numpy()
                ### fix timestamp position
                self.test_row_index = df_test.index.to_numpy()
            ### fix timestamp position
            df_train = df_train.reset_index(drop=True)
            ### fix timestamp position
            df_test = df_test.reset_index(drop=True)

        return df_train, df_test

    def feedback_dist(self, dataframe):
        """
        Transforms a dataframe into a new DataFrame where the columns are the
        categorical distribution probabilities of the target column values
        """
        df = dataframe.copy()
        dict_distrib = self.feedback_vector(df)
        df_distrib = self.distribution_feedback_vector(dict_distrib)
        df_distrib = df_distrib.set_index("user_id")

        return df_distrib

    def feedback_vector(self, dataframe):
        """Extracts one feedback vector for each user in `dataframe`"""
        df = dataframe.copy()
        user_id_list, dict_feedback = df["user_id"].unique(), {}

        for user_id in user_id_list:
            dict_feedback[user_id] = (
                df[df["user_id"] == user_id].loc[:, self.target_column].tolist()
            )

        return dict_feedback

    def distribution_feedback_vector(self, dict_feedback):
        """
        Calculates the probability distribution of all `classes` for each user.
        The classes are be sorted. E.g., [9, 10, 11]
        """
        df_distribution = pd.DataFrame()

        # for each user and its feedback
        for _, feedback in dict_feedback.items():
            nb_feedback = len(feedback)
            nb_classes = []

            # counts for each class value
            for label in self.target_values:
                nb_label = feedback.count(label)
                nb_classes.append(nb_label)

            # divide by total number of samples
            normalized_feedback = [x / nb_feedback for x in nb_classes]
            '''df_distribution = df_distribution.append(
                pd.Series(normalized_feedback), ignore_index=True
            )'''

            # create a new DataFrame from the Series to concat
            new_row = pd.DataFrame([normalized_feedback])

            df_distribution = pd.concat([df_distribution, new_row], ignore_index=True)


        # rename columns to the classes and insert user_ids
        df_distribution.columns = self.target_values
        df_distribution.insert(
            0, "user_id", [*dict_feedback]
        )  # unpack all the keys of the dict (user_ids)

        return df_distribution

    def cat_encoder(self, full=False):
        """Create encode object for list of categorical columns"""
        df = self.df_full.copy()
        df_qns = self.df_qns.copy()
        dict_le = {}

        # encode only `categorical_features`
        for col in self.categorical_features:
            dict_le[col] = LabelEncoder().fit(df[col])

        if full and self.qns_encoding_idx is not None:  # encode qnsfeatures too
            for col in list(self.df_qns.columns[self.qns_encoding_idx]):
                dict_le[col] = LabelEncoder().fit(df_qns[col])

        return dict_le

    def fit_cat_encoder(self, dataframe, full=False):
        """Fit the categorical encoder"""
        df = dataframe.copy()

        total_cat_features = (
            list(self.df_qns.columns[self.qns_encoding_idx]) + self.categorical_features
        )

        if (
            full and self.qns_encoding_idx is not None
        ):  # apply encoding to qns and cat features
            df = df.apply(
                lambda x: self.dict_encoder_full[x.name].transform(x)
                if x.name in total_cat_features
                else x
            )
            self.total_cat_features = total_cat_features
        else:  # apply encoding only to catfeatures
            df = df.apply(
                lambda x: self.dict_encoder[x.name].transform(x)
                if x.name in self.categorical_features
                else x
            )
        return df

    '''def onehot_encoder(self):
        """Create one-hot encode object for qns columns"""
        cat_columns = self.df_qns.iloc[:, self.qns_encoding_idx]  # encoding columns
        ohe = OneHotEncoder(categories=self.qns_categories, sparse=False)
        ohe = ohe.fit(cat_columns)

        return ohe'''
    
    def onehot_encoder(self):
        cat_columns = self.df_qns.iloc[:, self.qns_encoding_idx]
        ohe = OneHotEncoder(sparse_output=False)  # no categories arg
        ohe = ohe.fit(cat_columns)
        return ohe


    def fit_onehot_encoder(self, dataframe):
        """Fit one-hot encoder"""
        df = dataframe.copy()
        user_id_list = df.index.tolist()

        # remove categorical columns and encode
        df.drop(df.columns[self.qns_encoding_idx], axis=1, inplace=True)
        cat_columns = dataframe.iloc[:, self.qns_encoding_idx]
        cat_encoded = self.ohe.transform(cat_columns)

        # check if there are non-onehot categorical variables
        cat_qns_features = [
            value
            for value in self.categorical_features
            if value in self.df_qns.columns.values
        ]
        if cat_qns_features:  # if the list is not empty
            for cat_col in cat_qns_features:
                df[cat_col] = self.dict_encoder[cat_col].transform(df[cat_col])

        if self.gamma != 0.0:
            noise = np.random.uniform(0, self.gamma, cat_encoded.shape)
            cat_encoded = (cat_encoded + noise) / np.sum(
                cat_encoded + noise, keepdims=True, axis=1
            )

        # concatenate continuous + categorical
        df_concat = pd.DataFrame(
            np.concatenate((df, cat_encoded), axis=1), index=user_id_list
        )

        return df_concat
    
    def SMOTENC_resampling(self, df:  pd.DataFrame,class_name:str,SMOTENC_user:bool,SMOTENC_class:bool,SMOTENC_all_class:bool,class_limit_no_resampling:bool,class_min_resampling:int) -> pd.DataFrame:

        from imblearn.over_sampling import SMOTENC
        self.set_seed()
        print("Initialising data resampling ...")

        categorical_features = [0,4,5,6] #[0,3,4,5,6,7,8],

        if self.SMOTENC_all_class:
                    
            #### DO NOT CHECK THAT ENOUGH INSTANCES ARE IN THERE, can implement later, but now we know there are at least 5 in the whole dataset
            y = df.thermal_cozie
            X = df.drop(columns=["thermal_cozie"])

            sm = SMOTENC(categorical_features = categorical_features,random_state=self.seed,k_neighbors = class_min_resampling)
            X_res, y_res = sm.fit_resample(X, y)
            df_resampled = X_res
            df_resampled["thermal_cozie"] = y_res
            #this is so that the order of columns is the same for all df_resampled, no matter what version was used
            user_column = df_resampled["user_id"]
            df_resampled = df_resampled.drop(columns =["user_id"])
            df_resampled["user_id"] = user_column

        if self.SMOTENC_class:
            #target_column = 'thermal_cozie' #class_name
            per_user = True
            if per_user:
                df_resampled_list = []

                for id in df["user_id"].unique():

                    _df_id = df.loc[df.user_id == id,:]

                    num_counts = _df_id.thermal_cozie.value_counts() 
                    if class_limit_no_resampling: 
                        #if class_limit_no_resampling: if any class has not 5 entries: no resampling for that
                        # id, but we keep the data
                        if (len(num_counts[num_counts >= class_min_resampling]) < 3):#5
                            print("no resampling for id" + str(id))
                            _resampled = _df_id.drop(columns=["user_id","thermal_cozie"])
                            _resampled["thermal_cozie"] = _df_id.thermal_cozie
                            _resampled["user_id"] = id
                            df_resampled_list.append(_resampled)
                            continue

                    #if we want to resample as much as possible, or have enough entries in each class
                    _df_id = _df_id.loc[_df_id["thermal_cozie"].isin(num_counts[num_counts > class_min_resampling].index),:] #5

                    y = _df_id.thermal_cozie
                    X = _df_id.drop(columns=["user_id","thermal_cozie"])
                    # print(X[X.isna().any(axis=1)])
                    #print(df_sens.thermal_preference_num.value_counts().tail(10))
                    if len(_df_id.thermal_cozie.unique()) != 1:   
                        sm = SMOTENC(categorical_features = categorical_features,random_state=42,k_neighbors = class_min_resampling)
                        X_res, y_res = sm.fit_resample(X, y)
                        _resampled = X_res
                        _resampled["thermal_cozie"] = y_res
                    else:   
                        _resampled = X
                        _resampled["thermal_cozie"] = y
                    _resampled["user_id"] = id
                    df_resampled_list.append(_resampled)
            # print('Original dataset shape %s' % Counter(y))
            # print('Resampled dataset shape %s' % Counter(y_res))

                df_resampled = pd.concat(df_resampled_list)

                
        if self.SMOTENC_user:
            if self.SMOTENC_class:
                _df_id = df_resampled.copy()
            else:
                _df_id = df.copy()
            num_counts = _df_id.user_id.value_counts() 
            _df_id = _df_id.loc[_df_id["user_id"].isin(num_counts[num_counts > class_min_resampling].index),:]

            y = _df_id.user_id
            X = _df_id.drop(columns=["user_id"])
            # print(X[X.isna().any(axis=1)])
            #print(df_sens.thermal_preference_num.value_counts().tail(10))
            if len(_df_id.user_id.unique()) != 1:   
                sm = SMOTENC(categorical_features = categorical_features,random_state=42,k_neighbors = class_min_resampling)

                X_res, y_res = sm.fit_resample(X, y)
                _resampled = X_res
                _resampled["user_id"] = y_res
            else:
                _resampled = X
                _resampled["user_id"] = y
            df_resampled = _resampled

        return df_resampled
