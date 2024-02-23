def generate_instance_crossover_interpolation(X_train,X_explain, y_explain,indep,dep,blackbox_model, random_state=None, debug=False):
        """An approach to generate instance using Crossover and Interpolation

        Parameters
        ----------
        X_train : :obj:`pandas.core.frame.DataFrame`
            X_train (Training Features)
        X_explain : :obj:`pandas.core.frame.DataFrame`
            X_explain (Testing Features)
        y_explain : :obj:`pandas.core.series.Series`
            y_explain (Testing Label)
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

            # Generating instances using the cross-over technique
            for num in range(0, 1000):
                rand_rows = train_set_neigh.sample(2, random_state=random_state)
                sample_indexes_list = sample_indexes_list + rand_rows.index.values.tolist()
                # similarity_both = dist_df[dist_df.index.isin(rand_rows.index)]
                sample_classes = train_class_neigh[train_class_neigh.index.isin(
                    rand_rows.index)]
                sample_classes = np.array(
                    sample_classes.to_records().view(type=np.matrix))
                sample_classes_arr.append(sample_classes[0].tolist())

                alpha_n = np.random.uniform(low=0, high=1.0)
                x = rand_rows.iloc[0]
                y = rand_rows.iloc[1]
                new_ins = x + (y - x) * alpha_n
                new_ins = new_ins.to_frame().T

                """
                # For Categorical Variables
                for cat in categorical_vars:
                    x_df = x.to_frame().T
                    y_df = y.to_frame().T
                    # Check similarity of x > similarity of y
                    if similarity_both.iloc[0]['dist'] > similarity_both.iloc[1]['dist']:
                        new_ins[cat] = x_df.iloc[0][cat]
                    # Check similarity of y > similarity of x
                    elif similarity_both.iloc[0]['dist'] < similarity_both.iloc[1]['dist']:
                        new_ins[cat] = y_df.iloc[0][cat]
                    else:
                        new_ins[cat] = random.choice([x_df.iloc[0][cat], y_df.iloc[0][cat]])
                """
                new_ins.name = num
                new_con_df = new_con_df.append(new_ins, ignore_index=True)

            # Generating instances using the mutation technique
            for num in range(1000, 2000):
                rand_rows = train_set_neigh.sample(3, random_state=random_state)
                sample_indexes_list = sample_indexes_list + rand_rows.index.values.tolist()
                sample_classes = train_class_neigh[train_class_neigh.index.isin(
                    rand_rows.index)]
                sample_classes = np.array(
                    sample_classes.to_records().view(type=np.matrix))
                sample_classes_arr.append(sample_classes[0].tolist())
                mu_f = np.random.uniform(low=0.5, high=1.0)
                x = rand_rows.iloc[0]
                y = rand_rows.iloc[1]
                z = rand_rows.iloc[2]
                new_ins = x + (y - z) * mu_f
                new_ins = new_ins.to_frame().T
                """
                # For Categorical Variables get the value of the closest instance to the explained instance
                for cat in categorical_vars:
                    x_df = x.to_frame().T
                    y_df = y.to_frame().T
                    z_df = z.to_frame().T
                    new_ins[cat] = random.choice([x_df.iloc[0][cat], y_df.iloc[0][cat], z_df.iloc[0][cat]])
                """
                new_ins.name = num
                new_con_df = new_con_df.append(new_ins, ignore_index=True)

            # get the global model predictions of the generated instances and the instances in the neighbourhood
            predict_dataset = train_set_neigh.append(
                new_con_df, ignore_index=True)
            target = blackbox_model.predict(predict_dataset)
            target_df = pd.DataFrame(target)

            # neighbor_frequency = Counter(tuple(sorted(entry)) for entry in sample_classes_arr)

            new_df_case = pd.concat([predict_dataset, target_df], axis=1)
            new_df_case = np.round(new_df_case, 2)
            new_df_case.rename(columns={0: y_explain.columns[0]}, inplace=True)
            sampled_class_frequency = new_df_case.groupby([dep]).size()

            return {'synthetic_data': new_df_case,
                    'sampled_class_frequency': sampled_class_frequency}

    