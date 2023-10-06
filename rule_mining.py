import pandas as pd
import pandas as pd
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder


def apriori_generate_rule(bug_train_data, Notbug_train_data, T_min_support, F_min_support):
    te = TransactionEncoder()
    # print(bug_train_data)
    bug_train_data = te.fit_transform(bug_train_data.values.tolist())
    bug_train_data = pd.DataFrame(bug_train_data, columns=te.columns_)
    Tfrequent_itemsets = apriori(bug_train_data, min_support=T_min_support, use_colnames=True)
    Trules = association_rules(Tfrequent_itemsets, metric="support", min_threshold=T_min_support)

    fe = TransactionEncoder()
    Notbug_train_data = fe.fit_transform(Notbug_train_data.values.tolist())
    Notbug_test_data = pd.DataFrame(Notbug_train_data, columns=fe.columns_)
    Ffrequent_itemsets = apriori(Notbug_test_data, min_support=F_min_support, use_colnames=True)
    Frules = association_rules(Ffrequent_itemsets, metric="support", min_threshold=F_min_support)
    Trules["length"] = Trules["antecedents"].apply(lambda x: len(x))
    Frules["length"] = Frules["antecedents"].apply(lambda x: len(x))
    return Trules, Frules


def fpgrowth_generate_rule(bug_train_data, Notbug_train_data):
    te = TransactionEncoder()
    bug_train_data = te.fit_transform(bug_train_data.values.tolist())
    bug_train_data = pd.DataFrame(bug_train_data, columns=te.columns_)
    Tfrequent_itemsets = fpgrowth(bug_train_data, min_support=0.1, use_colnames=True)
    Trules = association_rules(Tfrequent_itemsets, metric="support", min_threshold=0.1)

    fe = TransactionEncoder()
    Notbug_train_data = fe.fit_transform(Notbug_train_data.values.tolist())
    Notbug_test_data = pd.DataFrame(Notbug_train_data, columns=fe.columns_)
    Ffrequent_itemsets = fpgrowth(Notbug_test_data, min_support=0.1, use_colnames=True)
    Frules = association_rules(Ffrequent_itemsets, metric="support", min_threshold=0.1)
    Trules["length"] = Trules["antecedents"].apply(lambda x: len(x))
    Frules["length"] = Frules["antecedents"].apply(lambda x: len(x))
    return Trules, Frules


def rank_rule(Trules, Frules, max_rules):
    Trules = Trules.sort_values(by=["support", "length"], ascending=False).reset_index(drop=True)
    Frules = Frules.sort_values(by=["support", "length"], ascending=False).reset_index(drop=True)
    if len(Trules) > max_rules:
        Trules = Trules.loc[:max_rules, :]
    if len(Frules) > max_rules:
        Frules = Frules.loc[:max_rules, :]
    return Trules, Frules


def remove_conflict_rule(Trules, Frules):
    columns = Trules.columns
    rules = pd.concat([Trules, Frules], axis=0)
    group = rules.groupby('antecedents')["consequents"].nunique().ne(1)
    rules = rules.loc[~rules['antecedents'].isin(group.index[group]), :].drop_duplicates().reset_index(drop=True)
    rulesT = []
    rulesF = []
    for index in range(len(rules)):
        if frozenset({"bug"}).issubset(rules.loc[index, "consequents"]):
            rulesT.append(rules.loc[index, :])

        if frozenset({"no bug"}).issubset(rules.loc[index, "consequents"]):
            rulesF.append(rules.loc[index, :])
    rulesT, rulesF = pd.DataFrame(rulesT, columns=columns).reset_index(drop=True), pd.DataFrame(rulesF,
                                                                                                columns=columns).reset_index(
        drop=True)
    return rulesT, rulesF


def remove_redundant_rule(Trules, Frules):
    drop_Tindex = []
    for i in range(len(Trules)):
        for j in range(i, len(Trules)):
            if ((frozenset(Trules.loc[i, "antecedents"]).issubset(
                    frozenset(Trules.loc[j, "antecedents"]))) & (
                    Trules.loc[i, "weightSupp"] > Trules.loc[j, "weightSupp"])):
                drop_Tindex.append(j)
            if ((frozenset(Trules.loc[j, "antecedents"]).issubset(
                    frozenset(Trules.loc[i, "antecedents"]))) & (
                    Trules.loc[j, "weightSupp"] > Trules.loc[i, "weightSupp"])):
                drop_Tindex.append(i)
    Trules.drop(drop_Tindex, axis=0, inplace=True)
    Trules.reset_index(drop=True)
    drop_Findex = []
    for i in range(len(Frules)):
        for j in range(i, len(Frules)):
            if ((frozenset(Frules.loc[i, "antecedents"]).issubset(
                    frozenset(Frules.loc[j, "antecedents"]))) & (
                    Frules.loc[i, "weightSupp"] > Frules.loc[j, "weightSupp"])):
                drop_Findex.append(j)
            if ((frozenset(Frules.loc[j, "antecedents"]).issubset(
                    frozenset(Frules.loc[i, "antecedents"]))) & (
                    Frules.loc[i, "weightSupp"] < Frules.loc[j, "weightSupp"])):
                drop_Findex.append(i)
    Frules.drop(drop_Findex, axis=0, inplace=True)
    Frules.reset_index(drop=True)
    return Trules, Frules


def rules_predict(test_data, combine_rule, evaluate):
    test_data["bug_proba"] = 0
    test_data["no_bug_proba"] = 0
    test_data["predict"] = 0
    test_data.loc[test_data["bug"] == "bug", "bug"] = 1
    test_data.loc[test_data["bug"] == "no bug", "bug"] = 0
    for data_index in range(len(test_data)):
        for index in range(len(combine_rule)):
            if combine_rule.loc[index, "antecedents"].issubset(test_data.loc[data_index, "combine"]):
                if "bug" in combine_rule.loc[index, "consequents"]:
                    test_data.loc[data_index, "bug_proba"] += combine_rule.loc[index, "weightSupp"]
                else:
                    test_data.loc[data_index, "CWCAR_no_bug_proba"] += combine_rule.loc[index, "weightSupp"]
        if test_data.loc[data_index, "no_bug_proba"] >= test_data.loc[data_index, "bug_proba"]:
            test_data.loc[data_index, "predict"] = 0
        else:
            test_data.loc[data_index, "CWCAR_predict"] = 1
    test_data["bug"] = test_data["bug"].astype("float")
    CWCAR_matrix = confusion_matrix(test_data["bug"], test_data["predict"])
    CWCAR_balance, CWCAR_gmean, CWCAR_mcc, CWCAR_Acc, CWCAR_Recall, CWCAR_f1 = caculate_eval(
                                                                                             test_data["bug"],
                                                                                             test_data["CWCAR_predict"])
    evaluate["Balance"].append(balance)
    evaluate["Gmean"].append(gmean)
    evaluate["Mcc"].append(mcc)
    evaluate["Acc"].append(Acc)
    evaluate["Recall"].append(Recall)
    evaluate["f1"].append(f1)
    return evaluate


def model(train_data, test_data, featureWeight, fold, evaluate filename, T_min_support, F_min_support):
    new_train_data, new_test_data = discretion(train_data, test_data)
    new_TSet, new_FSet = split_TF2(new_train_data)
    Trules, Frules = apriori_generateRule(new_TSet, new_FSet, T_min_support, F_min_support)
    Trules, Frules = rank(Trules, Frules, 20000)
    Trules, Frules = conflict(Trules, Frules)
    Trules["weightSupp"] = 1
    Frules["weightSupp"] = 1
    T_dropIndex = []
    F_dropIndex = []
    for k in range(len(Trules)):
        if ("bug" in Trules.loc[k, "antecedents"] or "bug" not in \
                Trules.loc[k, "consequents"]):
            T_dropIndex.append(k)
    Trules = Trules.drop(T_dropIndex, axis=0)
    for z in range(len(Frules)):
        if ("no bug" in Frules.loc[z, "antecedents"] or "no bug" not in \
                Frules.loc[z, "consequents"]):
            F_dropIndex.append(z)
    Frules = Frules.drop(F_dropIndex, axis=0)
    for Tindex in range(len(Trules)):
        if len(Trules.loc[Tindex, "antecedents"]) == 1:
            for feature in Trules.loc[Tindex, "antecedents"]:
                Trules.loc[Tindex, "weightSupp"] = Trules.loc[Tindex, "support"] * featureWeight[feature.split("=")[0]]
        else:
            Trules.loc[Tindex, "weightSupp"] = Trules.loc[Tindex, "support"]
            for key in Trules.loc[Tindex, "antecedents"]:
                Trules.loc[Tindex, "weightSupp"] *= featureWeight[
                    key.split("=")[0]]
    for Findex in range(len(Frules)):
        if len(Frules.loc[Findex, "antecedents"]) == 1:
            for feature in Frules.loc[Findex, "antecedents"]:
                Frules.loc[Findex, "weightSupp"] = Frules.loc[Findex, "support"] * featureWeight[feature.split("=")[0]]
        else:
            Frules.loc[Findex, "weightSupp"] = Frules.loc[Findex, "support"]
            for key in Frules.loc[Findex, "antecedents"]:
                Frules.loc[Findex, "weightSupp"] *= featureWeight[key.split("=")[0]]
    Trules = Trules.drop_duplicates(['antecedents'], keep='first')
    Frules = Frules.drop_duplicates(['antecedents'], keep='first')
    Trules.to_csv(f"./rules/T/{fold + 1}{filename}", index=False)
    Frules.to_csv(f"./rules/F/{fold + 1} {filename}", index=False)
    # remove rules
    Trules, Frules = redundant(Trules, Frules)
    Trules.to_csv(f"./rules/T/{fold + 1}{filename}", index=False)
    Frules.to_csv(f"./rules/F/{fold + 1} {filename}", index=False)
    # combine datasets
    test_data = data_combine_test(test_data)
    test_data.to_csv("test_data.csv", index=False)
    combine_rule = pd.concat([Trules, Frules], axis=0).reset_index(drop=True)
    combine_rule.to_csv(f"./rules/combine_rule/{fold + 1}{filename}", index=False)
    evaluate = rules_predict(test_data, combine_rule, evaluate)
    print(
        f'{fold + 1}fold CWCAR_Balance：{evaluate["CWCAR_Balance"][fold]},CWCAR_Mcc:  {evaluate["CWCAR_Mcc"][fold]},CWCAR_Gmean:{evaluate["CWCAR_Gmean"][fold]}')
    return evaluate