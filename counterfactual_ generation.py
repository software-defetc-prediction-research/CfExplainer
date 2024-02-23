import os
import dice_ml
import pandas as pd
from pyexplainer import pyexplainer_pyexplainer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, train_test_split
import warnings
from sklearn.metrics.pairwise import euclidean_distances


def generate_instance_counterfactual(X_train,X_explain, y_explain,indep,dep,blackbox_model,pretain_model, random_state=None, debug=False):
        """An approach to generate instance using Crossover and Interpolation

        Parameters
        ----------
        X_train : :obj:`pandas.core.frame.DataFrame`
            X_train (Training Features)
        X_explain : :obj:`pandas.core.frame.DataFrame`
            X_explain (Testing Features)
        y_explain : :obj:`pandas.core.series.Series`
            y_explain (Testing Label)
        blackbox_model:blackbox model
        pretain_model:Pre-training model
        random_state : :obj:`int`
            Random Seed
        debug : :obj:`bool`
            True for debugging mode, False otherwise.

        Returns
        -------
        :obj:`dict`
            A dict with two keys 'synthetic_data' and 'sampled_class_frequency' generated via Crossover and Interpolation.
        """
        # categorical_vars = []

        X_train_i = X_train.copy()
        X_explain = X_explain.copy()
        y_explain = y_explain.copy()

        X_train_i.reset_index(inplace=True)
        X_explain.reset_index(inplace=True)
        X_train_i = X_train_i.loc[:, indep]

        X_explain = X_explain.loc[:, indep]
        y_explain = y_explain.reset_index()[[dep]]

        # get the global model predictions for the training set
        target_train = blackbox_model.predict(X_train_i)


        # Do feature scaling for continuous data and one hot encoding for categorical data
        scaler = StandardScaler()
        trainset_normalize = X_train_i.copy()
        if debug:
            print(list(X_train_i), "columns")
        cases_normalize = X_explain.copy()

        train_objs_num = len(trainset_normalize)
        dataset = pd.concat(objs=[trainset_normalize, cases_normalize], axis=0)
        if debug:
            print(indep, "continuous")
            print(type(indep))
        dataset[indep] = scaler.fit_transform(dataset[indep])
        trainset_normalize = copy.copy(dataset[:train_objs_num])
        cases_normalize = copy.copy(dataset[train_objs_num:])

        # make dataframe to store similarities of the trained instances from the explained instance
        dist_df = pd.DataFrame(index=trainset_normalize.index.copy())

        width = math.sqrt(len(X_train_i.columns)) * 0.75
        # similarity
        for count, case in cases_normalize.iterrows():
            # Calculate the euclidean distance from the instance to be explained
            dist = np.linalg.norm(
                trainset_normalize.sub(np.array(case)), axis=1)
            # Convert distance to a similarity score
            similarity = np.exp(-(dist ** 2) / (2 * (width ** 2)))
            dist_df['dist'] = similarity
            dist_df['t_target'] = target_train
            # get the unique classes of the training set
            unique_classes = dist_df.t_target.unique()
            # Sort similarity scores in to descending order
            dist_df.sort_values(by=['dist'], ascending=False, inplace=True, kind='mergesort')
            # dist_df.reset_index(inplace=True)

            # Make a dataframe with top 40 elements in each class
            top_fourty_df = pd.DataFrame([])
            for clz in unique_classes:
                top_fourty_df = top_fourty_df.append(
                    dist_df[dist_df['t_target'] == clz].head(40))

            # get the minimum value of the top 40 elements and return the index
            cutoff_similarity = top_fourty_df.nsmallest(
                1, 'dist', keep='last').index.values.astype(int)[0]

            # Get the location for the given index with the minimum similarity
            min_loc = dist_df.index.get_loc(cutoff_similarity)
            # whole neighbourhood without undersampling the majority class
            train_neigh_sampling_b = dist_df.iloc[0:min_loc + 1]
            # get the size of neighbourhood for each class
            target_details = train_neigh_sampling_b.groupby(
                ['t_target']).size()
            if debug:
                print(target_details, "target_details")
            target_details_df = pd.DataFrame(
                {'target': target_details.index, 'target_count': target_details.values})

            # Get the majority class and undersample
            final_neighbours_similarity_df = pd.DataFrame([])
            for index, row in target_details_df.iterrows():
                if row["target_count"] > 200:
                    filterd_class_set = train_neigh_sampling_b \
                        .loc[train_neigh_sampling_b['t_target'] == row['target']] \
                        .sample(n=200, random_state=random_state)
                    final_neighbours_similarity_df = final_neighbours_similarity_df.append(
                        filterd_class_set)
                else:
                    filterd_class_set = train_neigh_sampling_b \
                        .loc[train_neigh_sampling_b['t_target'] == row['target']]
                    final_neighbours_similarity_df = final_neighbours_similarity_df.append(
                        filterd_class_set)
            if debug:
                print(final_neighbours_similarity_df,
                      "final_neighbours_similarity_df")
            # Get the original training set instances which is equal to the index of the selected neighbours
            train_set_neigh = X_train_i[X_train_i.index.isin(
                final_neighbours_similarity_df.index)]
            if debug:
                print(train_set_neigh, "train set neigh")
            train_class_neigh = y_explain[y_explain.index.isin(
                final_neighbours_similarity_df.index)]


            new_con_df = pd.DataFrame([])
            sample_classes_arr = []
            sample_indexes_list = []
            # Loading Data
            d = dice_ml.Data(dataframe=X_train, continuous_features=X_train.drop("buggy").columns.tolist(), outcome_name='buggy')
            # Loading counterfactual models
            m = dice_ml.Model(model=pretain_model, backend="sklearn")
            exp = dice_ml.Dice(d, m, method="random")
            synthetic_instance = pd.DataFrame()
            # Generating instances using the counterfactual
            for i in X_train.index:
                counterfactuals = exp.generate_counterfactuals(X_train.iloc[i:i + 1, :], total_CFs=1,desired_class="opposite")
                cf = counterfactuals.cf_examples_list[0].final_cfs_df
                counterfactuals.visualize_as_dataframe(show_only_changes=True)
                synthetic_instance = synthetic_instance.append(cf, ignore_index=True) 
                if len(synthetic_instance)>2000:
                    break

           
            # get the global model predictions of the generated instances and the instances in the neighbourhood
            predict_dataset = train_set_neigh.append(
                synthetic_instance, ignore_index=True)
            target = blackbox_model.predict(predict_dataset)
            target_df = pd.DataFrame(target)

            # neighbor_frequency = Counter(tuple(sorted(entry)) for entry in sample_classes_arr)

            new_df_case = pd.concat([predict_dataset, target_df], axis=1)
            new_df_case = np.round(new_df_case, 2)
            new_df_case.rename(columns={0: y_explain.columns[0]}, inplace=True)
            sampled_class_frequency = new_df_case.groupby([dep]).size()

            return {'synthetic_data': new_df_case,
                    'sampled_class_frequency': sampled_class_frequency}

    