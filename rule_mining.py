import pandas as pd
import pandas as pd
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder
from sklearn.metrics import confusion_matrix

from data_handale import discretion, split_TF2, data_combine_test
from evaluation import caculate_eval


class rule_minning:
    """
    Init method
    :param T_min_support: Minimum support for defective rules
    :param F_min_support: Minimum support for non-defective rules
    :param max_rule: Maximum number of rules
    """

    def __init__(self,
                 T_min_support,
                 F_min_support,
                 max_rule):
        self.max_rule = max_rule
        self.T_min_support = T_min_support
        self.F_min_support = F_min_support

    """
    Mining weighted class association rules using apriori algorithm
    :param bug_train_data: defective data
    :param Notbug_train_data: non-defective data
    """

    def apriori_generate_rule(self, bug_train_data, Notbug_train_data):
        te = TransactionEncoder()
        # print(bug_train_data)
        bug_train_data = te.fit_transform(bug_train_data.values.tolist())
        bug_train_data = pd.DataFrame(bug_train_data, columns=te.columns_)
        Tfrequent_itemsets = apriori(bug_train_data, min_support=self.T_min_support, use_colnames=True)
        Trules = association_rules(Tfrequent_itemsets, metric="support", min_threshold=self.T_min_support)

        fe = TransactionEncoder()
        Notbug_train_data = fe.fit_transform(Notbug_train_data.values.tolist())
        Notbug_test_data = pd.DataFrame(Notbug_train_data, columns=fe.columns_)
        Ffrequent_itemsets = apriori(Notbug_test_data, min_support=self.F_min_support, use_colnames=True)
        Frules = association_rules(Ffrequent_itemsets, metric="support", min_threshold=self.F_min_support)
        Trules["length"] = Trules["antecedents"].apply(lambda x: len(x))
        Frules["length"] = Frules["antecedents"].apply(lambda x: len(x))
        return Trules, Frules

    """
    Mining weighted class association rules using fpgrowth algorithm
    :param bug_train_data: defective data
    :param Notbug_train_data: non-defective data
    """

    def fpgrowth_generate_rule(self, bug_train_data, Notbug_train_data):
        te = TransactionEncoder()
        bug_train_data = te.fit_transform(bug_train_data.values.tolist())
        bug_train_data = pd.DataFrame(bug_train_data, columns=te.columns_)
        Tfrequent_itemsets = fpgrowth(bug_train_data, min_support=0.1, use_colnames=True)
        Trules = association_rules(Tfrequent_itemsets, metric="support", min_threshold=self.T_min_support)

        fe = TransactionEncoder()
        Notbug_train_data = fe.fit_transform(Notbug_train_data.values.tolist())
        Notbug_test_data = pd.DataFrame(Notbug_train_data, columns=fe.columns_)
        Ffrequent_itemsets = fpgrowth(Notbug_test_data, min_support=0.1, use_colnames=True)
        Frules = association_rules(Ffrequent_itemsets, metric="support", min_threshold=self.F_min_support)
        Trules["length"] = Trules["antecedents"].apply(lambda x: len(x))
        Frules["length"] = Frules["antecedents"].apply(lambda x: len(x))
        return Trules, Frules

    """
    order by rule
    :param Trules: defective rule
    :param Frules: non-defective rule
    """

    def rank_rule(self, Trules, Frules):
        Trules["generate_time"] = range(1, len(Trules) + 1)
        Frules["generate_time"] = range(1, len(Frules) + 1)
        Trules = Trules.sort_values(by=["support", "length", "generate_time"], ascending=False).reset_index(drop=True)
        Frules = Frules.sort_values(by=["support", "length", "generate_time"], ascending=False).reset_index(drop=True)
        if len(Trules) > self.max_rules:
            Trules = Trules.loc[:self.max_rules, :]
        if len(Frules) > self.max_rules:
            Frules = Frules.loc[:self.max_rules, :]
        return Trules, Frules

    """
    Removing conflicting  rules
    :param Trules: defective rule
    :param Frules: non-defective rule
    """

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

    """
    Removing redundant rules
    :param Trules: defective rule
    :param Frules: non-defective rule
    """

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

    """
    Multi-Rule Prediction
    :param test_data: test data
    :param combine_rule: Rules for combining
    """

    def multi_rule_predict(test_data, combine_rule):
        test_data["bug_proba"] = 0
        test_data["no_bug_proba"] = 0
        test_data["predict"] = 0
        test_data.loc[test_data["buggy"] == "buggy", "buggy"] = 1
        test_data.loc[test_data["buggy"] == "no buggy", "buggy"] = 0
        for data_index in range(len(test_data)):
            for index in range(len(combine_rule)):
                if combine_rule.loc[index, "antecedents"].issubset(test_data.loc[data_index, "combine"]):
                    if "buggy" in combine_rule.loc[index, "consequents"]:
                        test_data.loc[data_index, "bug_proba"] += combine_rule.loc[index, "weightSupp"]
                    else:
                        test_data.loc[data_index, "CWCAR_no_bug_proba"] += combine_rule.loc[index, "weightSupp"]
            if test_data.loc[data_index, "no_bug_proba"] >= test_data.loc[data_index, "bug_proba"]:
                test_data.loc[data_index, "predict"] = 0
            else:
                test_data.loc[data_index, "CWCAR_predict"] = 1
        test_data["buggy"] = test_data["buggy"].astype("float")
        return test_data["buggy"]
