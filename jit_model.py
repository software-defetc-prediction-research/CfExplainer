import os

import dice_ml
import pandas as pd
from scipy.stats import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from pyexplainer import pyexplainer_pyexplainer

path = "./data/"
# filename = "qt.csv"
columns = ["commit_id", "author_date", "fix", "bugcount", "fixcount"]


def load_data(path, filename):
    # load data
    data = pd.read_csv(path + filename)
    # Sort the data
    data['author_date'] = pd.to_datetime(data['author_date'])
    # Sort by author_date column in ascending order
    data = data.sort_values(by='author_date')
    return data


# Data preprocessing
def data_preprocessing(data):
    data.drop(columns=columns, inplace=True)
    data = data.dropna(axis=1, how='all')
    data.loc[data["buggy"] == False, "buggy"] = 0
    data.loc[data["buggy"] == True, "buggy"] = 1
    data.loc[data["revd"] == False, "revd"] = 0
    data.loc[data["revd"] == True, "revd"] = 1
    data.loc[data["self"] == False, "self"] = 0
    data.loc[data["self"] == True, "self"] = 1
    data['buggy'] = data['buggy'].astype(int)
    data = data.dropna(how="any")
    return data


# feature select
def feature_select(data, correlation_threshold=0.7, ):
    data.to_csv("data.csv", index=False)
    data = pyexplainer_pyexplainer.AutoSpearman(data, correlation_threshold=correlation_threshold,
                                                correlation_method='spearman')
    return data


# data imbalance handle
def data_balance(data):
    # count difference
    bug_number = len(data[data["buggy"] == 1])
    clean_number = len(data[data["buggy"] == 0])
    difference = abs(clean_number - bug_number)

    bug_data = data[data["buggy"] == 1]
    clean_data = data[data["buggy"] == 1]
    feature = data.columns.tolist().remove("bug")

    model = RandomForestClassifier()
    model.fit(data.drop(columns="buggy"), data["buggy"])

    # load data
    d = dice_ml.Data(dataframe=data, continuous_features=feature, outcome_name='buggy')
    # load counterfactuals model
    m = dice_ml.Model(model=model, backend="sklearn")

    # genereta counterfactuals instances
    exp = dice_ml.Dice(d, m, method="genetic")
    # exp = dice_ml.Dice(d, m, method="random")
    index = bug_data.index
    for i in index:
        counterfactuals = exp.generate_counterfactuals(bug_data.iloc[i, :], total_CFs=1,
                                                       desired_class="opposite")
        cf = counterfactuals.cf_examples_list[0].final_cfs_df
        counterfactuals.visualize_as_dataframe(show_only_changes=True)
        bug_data = bug_data.append(cf, ignore_index=True)
    data = bug_data.append(clean_data, ignore_index=True)
    return data


def split_data(data):
    # Delineate training and test sets
    X_train, X_test, y_train, y_test = train_test_split(data.drop(columns="buggy"), data["buggy"], test_size=0.3,
                                                        shuffle=False)
    for feature in X_train.columns:
        print(f"特征{feature}")
        result = stats.normaltest(X_train[feature])
        print("Shapiro-Wilk统计量值:", result[0])
        print("p-value:", result[1])
    return X_train, X_test, y_train, y_test


def RF_model_train(X_train, y_train):
    blackbox_model = RandomForestClassifier()
    blackbox_model.fit(X_train, y_train)
    return blackbox_model


def LR_model_train(X_train, y_train):
    blackbox_model = RandomForestClassifier()
    blackbox_model.fit(X_train, y_train)
    return blackbox_model


def caculate_eval(label, predicted, y_prob):
    f1 = f1_score(label, predicted)
    auc = roc_auc_score(label, y_prob)
    return f1, auc


if __name__ == '__main__':
    for filename in os.listdir(path):
        # load data
        print(f"读取文件{filename}")
        data = load_data(path, filename)
        # data process
        data = data_preprocessing(data)
        # Delineate training and test sets
        X_train, X_test, y_train, y_test = split_data(data)
        test_data = pd.concat([X_train, y_train], axis=1)
        # print(test_data.info())
        # print(f"缺陷数量={len(test_data[test_data['buggy'] == 1])}", f"非缺陷数量={len(test_data[test_data['buggy'] == 0])}",
        #       f"缺陷数量={len(test_data[test_data['buggy'] == 1]) /len(test_data[test_data['buggy']==0])}")

        # static_data = pd.DataFrame()
        # for column in test_data.columns:
        #     # print(f"众数={test_data[column].mode()}")
        #     # print(f"中位数={test_data[column].median()}")
        #     # print(f"均值={test_data[column].mean()}")
        #     # static_data[f"{column}众数"] = test_data[column].mode()
        #     static_data[f"{column}中位数"] = test_data[column].median()
        #     static_data[f"{column}均值"] = test_data[column].mean()
        # static_data.to_csv(f"static_{filename}", index=False)
        # data combine
        train_data = pd.concat([X_train, y_train], axis=1)
        # # feature select
        train_data = feature_select(train_data)
        # # data imbalance handle
        # train_data = data_balance(train_data)
        # # model train process
        # RF_global_model = RF_model_train(train_data.drop(columns="buggy"), train_data["buggy"])
        # LR_global_model = LR_model_train(train_data.drop(columns="buggy"), train_data["buggy"])
        # # model predict process
        # RF_predicted = RF_global_model.predict(X_test, y_test)
        # RF_y_prob = RF_global_model.predict_proba(X_test)
        # LR_predicted = LR_global_model.predict(X_test, y_test)
        # LR_y_prob = LR_global_model.predict_proba(X_test)
        # # caculat metric
        # f1, auc = caculate_eval(y_test, RF_predicted, RF_y_prob)
        # f1, auc = caculate_eval(y_test, LR_predicted, LR_y_prob)
