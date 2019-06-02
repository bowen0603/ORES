from parser_adult import Adult
from parser_dutch import Dutch
from parser_lawschool import Lawschool
from parser_campus import Compas
from parser_wiki import ParserWiki

import sys
from sklearn.preprocessing import scale
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import fairlearn.moments as moments
import fairlearn.classred as red
from sklearn import cross_validation as cv
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import operator

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve

__author__ = 'bobo'


class PredictionFairness:

    def __init__(self, EO):
        self.decimal = 3
        self.n_folds = 1
        self.N = 19412
        self.eps = 0.100
        self.threshold_density = 41  #21
        # Disparity FPR: 0.015
        # self.list_eps = [0.001, 0.0025, 0.005, 0.0075, 0.01, 0.0125, 0.015, 0.0175, 0.02, 0.025, 0.03, 0.04,
        #                  0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.2, 0.3, 0.4, 0.5]
        self.list_eps = [-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        self.list_eps = np.linspace(-1, 1, 81, endpoint=True)
        # self.list_eps = np.linspace(-3, 3, 61, endpoint=True)

        # self.list_eps = [range(0.01, 0.2, 0.01)]
        # self.list_eps = np.arange(0.01, 0.2, 0.01)
        # self.list_eps = [0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.3, 0.5]
        # self.list_eps = [0.001, 0.01, 0.03, 0.05, 0.075, 0.1, 0.3, 0.5]
        # self.list_eps = [0.0001, 0.00025, 0.0005, 0.00075, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05]

        self.data_x = None
        self.data_y = None
        self.data_x_protected = None
        self.data_y_badfaith = None
        self.data_y_damaging = None
        self.data_x_g0 = []
        self.data_y_g0 = []
        self.data_x_g1 = []
        self.data_y_g1 = []
        self.df = None

        self.label_type = 'quality'
        self.plot_output = 'dataset/plot_data_fairness'
        self.EO = EO

    def load_data(self):
        reader = ParserWiki()
        self.df = reader.parse_data()

        # TODO: no rescaling on boolean variables..
        # self.data_x = self.data_rescale(reader.data_x)
        self.data_x = reader.data_x
        self.data_y_damaging = reader.data_y_damaging
        self.data_y_badfaith = reader.data_y_badfaith

    def retrieve_editor_info(self):
        pass

    def data_rescale(self, data):
        return scale(data)

    # Data Summary on registered v.s. unregistered editors
    # unregistered, damaging: 481; unregistered, good: 3007
    # registered, damaging: 270; registered, good: 15654
    def data_reformulation(self):

        data_adjusted_x, data_adjusted_y = [], []
        cnt_anon_pos, cnt_anon_neg, cnt_reg_pos, cnt_reg_neg = 0, 0, 0, 0

        if self.label_type == 'quality':
            self.data_y = self.data_y_damaging
        elif self.label_type == 'intention':
            self.data_y = self.data_y_badfaith
        else:
            return

        for i in range(len(self.data_x)):

            if self.data_y[i] == 1 and self.data_x[i][0] == 1:
                cnt_anon_pos += 1
            if self.data_y[i] == 0 and self.data_x[i][0] == 1:
                cnt_anon_neg += 1
            if self.data_y[i] == 1 and self.data_x[i][0] == 0:
                cnt_reg_pos += 1
            if self.data_y[i] == 0 and self.data_x[i][0] == 0:
                cnt_reg_neg += 1

            if self.EO == 'FPR':
                # Case 1: equalize FP rates
                # Make all the positive examples with y=1 belong to the same group (a = 1)
                if self.data_y[i] == 1 and self.data_x[i][0] == 1:
                    data_adjusted_x.append(self.data_x[i])
                    data_adjusted_y.append(self.data_y[i])
                if self.data_y[i] == 0:
                    data_adjusted_x.append(self.data_x[i])
                    data_adjusted_y.append(self.data_y[i])

            elif self.EO == 'FNR':
                # Case 2: equalize FN rates
                # Make all the negative examples with y=0 belong to the same group (a = 1)
                if self.data_y[i] == 0 and self.data_x[i][0] == 1:
                    data_adjusted_x.append(self.data_x[i])
                    data_adjusted_y.append(self.data_y[i])
                if self.data_y[i] == 1:
                    data_adjusted_x.append(self.data_x[i])
                    data_adjusted_y.append(self.data_y[i])

            elif self.EO == 'BOTH':
                # Case 3: equalize both FP and FN rates
                data_adjusted_x.append(self.data_x[i])
                data_adjusted_y.append(self.data_y[i])

            # Collect data for two groups
            # TODO: sanity check on these two datasets ..
            if self.data_x[i][0] == 0:
                self.data_x_g0.append(self.data_x[i])
                self.data_y_g0.append(self.data_y[i])

            if self.data_x[i][0] == 1:
                self.data_x_g1.append(self.data_x[i])
                self.data_y_g1.append(self.data_y[i])

        # Adjusted data
        self.data_x = np.array(data_adjusted_x)
        self.data_y = pd.Series(data_adjusted_y)

        self.data_sanity_check(self.data_x, self.data_y)

        print("Attr 0 (registered): {}, Attr 1: {} (unregistered).".format(cnt_reg_pos + cnt_reg_neg,
                                                                           cnt_anon_pos + cnt_anon_neg))
        print("{}+{}, {}+{}".format(cnt_reg_pos, cnt_reg_neg, cnt_anon_pos, cnt_anon_neg))

        # extract the column of the protected attribute (convert to string ..)
        self.data_x_protected = pd.Series(self.data_x[:, [0]].tolist()).apply(str)
        # delete the column of the protected attribute
        # self.data_x = np.delete(self.data_x, 0, 1)

    @staticmethod
    def data_sanity_check(data_x, data_y):
        cnt_pos_g0, cnt_pos_g1, cnt_neg_g0, cnt_neg_g1 = 0, 0, 0, 0
        for idx in range(len(data_x)):
            if data_y[idx] == 1:
                if data_x[idx][0] == 0:
                    cnt_pos_g0 += 1
                else:
                    cnt_pos_g1 += 1
            else:
                if data_x[idx][0] == 0:
                    cnt_neg_g0 += 1
                else:
                    cnt_neg_g1 += 1
        print("pos g0: {}, pos g1: {}, neg g0: {}, neg g1: {}".format(cnt_pos_g0, cnt_pos_g1, cnt_neg_g0, cnt_neg_g1))

    @staticmethod
    def split_train_test_data(data_x, data_y=None):
        indices = np.arange(len(data_x))

        data_x = np.array(data_x)
        data_x = np.delete(data_x, 0, 1)
        data_y = np.array(data_y)
        # todo: check if it returns the same idx in each call ..
        X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(data_x, data_y,
                                                                                 indices,
                                                                                 test_size=0.3, random_state=22)
        X_train, X_test = data_x[train_idx], data_x[test_idx]
        y_train, y_test = data_y[train_idx], data_y[test_idx]
        return X_train, X_test, y_train, y_test, train_idx

    @staticmethod
    def collect_classifiers():
        return [LogisticRegression(), RandomForestClassifier(), GradientBoostingClassifier()]


    def run_train_test_split_baseline2(self):
        # Adult(), Compas(), Dutch(), Lawschool(), ParserWiki()
        for obj in [ParserWiki()]:

            train, test, idx_X, idx_A, idx_y, data_name = obj.create_data()
            print(data_name, train.shape, test.shape, len(idx_X), len(idx_A), len(idx_y))

            # train the model using the train data with full features
            # clf = GradientBoostingClassifier(learning_rate=0.01, max_depth=7, max_features="sqrt", n_estimators=700)
            # clf = RandomForestClassifier()
            clf = LogisticRegression(max_iter=500, penalty='l1', C=1)  # default P>0.5
            # clf.fit(train[idx_X+idx_A], train[idx_y])
            clf.fit(train[idx_X + idx_A], train[idx_y])

            df_X = test[idx_X+idx_A]
            X_ga0 = df_X.loc[df_X[idx_A[0]] == 0]
            X_ga1 = df_X.loc[df_X[idx_A[0]] == 1]

            df_y = test[idx_y]
            y_ga0 = df_y.loc[df_X[idx_A[0]] == 0]
            y_ga1 = df_y.loc[df_X[idx_A[0]] == 1]

            y_pred_prob = clf.predict_proba(test[idx_X+idx_A])
            y_pred_score = []
            for neg, pos in y_pred_prob:
                y_pred_score.append(pos)

            # list_roc_auc_damaging.append(roc_auc_score(y_true=y_test_damaging, y_score=y_pred_score))
            # list_pr_auc_damaging.append(average_precision_score(y_true=y_test_damaging, y_score=y_pred_score))
            #
            # list_roc_auc_badfaith.append(roc_auc_score(y_true=y_test_badfaith, y_score=y_pred_score))
            # list_pr_auc_badfaith.append(average_precision_score(y_true=y_test_badfaith, y_score=y_pred_score))

            # clf.fit(test[idx_X+idx_A], train[idx_y])
            # y_pred = clf.predict_proba(test[idx_X+idx_A])[:, 1]
            # logistic regression: 0.043 0.246 (without protected attribute)  # 0.074 0.244 (with protected attribute)
            # gradient boosting: 0.063 0.177 (with protected attribute)
            # y_pred = clf.predict(test[idx_X+idx_A])
            # 0.025 0.176  # 0.044 0.173
            tn, fp, fn, tp = confusion_matrix(y_ga0, clf.predict(X_ga0), [0, 1]).ravel()
            print(tn, fp, fn, tp)
            print(fp / (fp+tn))
            tn, fp, fn, tp = confusion_matrix(y_ga1, clf.predict(X_ga1), [0, 1]).ravel()
            print(tn, fp, fn, tp)
            print(fp / (fp + tn))

            # error_train = self.compute_error(train[idx_y[0]], y_pred)
            # disparity_train = self.compute_FP(train[idx_A[0]], train[idx_y[0]], y_pred)
            #
            # print(disparity_train, error_train)

            single_test = True
            if single_test:
                return

            thresholds = np.linspace(0, 1, self.threshold_density, endpoint=True)
            print(len(y_pred_prob))
            filename='dataset/plot.csv'
            f_output = open(filename, 'w')
            for threshold in thresholds:
                y_prob = y_pred_prob[:, 1]
                y_prob[y_prob >= threshold] = 1
                y_prob[y_prob < threshold] = 0

                error_train = self.compute_error(train[idx_y[0]], y_prob)
                disparity_train = self.compute_FP(train[idx_A[0]], train[idx_y[0]], y_prob)

                print("{},{},{}".format(threshold, disparity_train, error_train))
                print("{},{},{}".format(threshold, disparity_train, error_train), file=f_output)

            f_output.close()
            self.plot_charts(filename)

    def run_train_test_split_baseline(self):

        X_train_g0, X_test_g0, y_train_g0, y_test_g0, _ = self.split_train_test_data(self.data_x_g0, self.data_y_g0)
        X_train_g1, X_test_g1, y_train_g1, y_test_g1, _ = self.split_train_test_data(self.data_x_g1, self.data_y_g1)
        X_train_g01, X_test_g01, y_train_g01, y_test_g01, _ = self.split_train_test_data(
            np.array(self.data_x_g0 + self.data_x_g1),
            np.array(self.data_y_g0 + self.data_y_g1))

        clf = LogisticRegression()
        clf.fit(X_train_g01, y_train_g01)
        thresholds = np.linspace(0, 1, 101, endpoint=True)

        # Predict on the two groups
        y_pred_prob_g0 = clf.predict_proba(X_test_g0)
        y_pred_prob_g1 = clf.predict_proba(X_test_g1)

        f_output_train = open("{}_{}_train.csv".format(self.plot_output, self.label_type), 'w')

        for threshold in thresholds:
            list_y_pred_g0, list_y_pred_g1 = [], []
            for score in y_pred_prob_g0:
                list_y_pred_g0.append(1 if score[1] >= threshold else 0)
            for score in y_pred_prob_g1:
                list_y_pred_g1.append(1 if score[1] >= threshold else 0)

            tn0, fp0, fn0, tp0 = confusion_matrix(y_true=y_test_g0, y_pred=list_y_pred_g0).ravel()
            tn1, fp1, fn1, tp1 = confusion_matrix(y_true=y_test_g1, y_pred=list_y_pred_g1).ravel()

            disparity_fpr = abs(fp0 / (fp0 + tn0) - fp1 / (fp1 + tn1))
            disparity_fnr = abs(fn0 / (fn0 + tp0) - fn1 / (fn1 + tp1))
            error_rate = (fp0 + fn0 + fp1 + fn1) / (tp0 + tn0 + fp0 + fn0 + tp1 + tn1 + fp1 + fn1)

            print("{},{},{}".format(threshold, disparity_fpr, error_rate), file=f_output_train)
            print("{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}".format(threshold,
                                                          disparity_fpr, disparity_fnr,
                                                          error_rate))

    def run_train_test_split_fairlearn(self):
        # Train the model using adjusted data
        indices = np.arange(len(self.data_x))
        X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(self.data_x, self.data_y, indices,
                                                                                 test_size=0.3, random_state=12)

        X_train, X_test = self.data_x[train_idx], self.data_x[test_idx]
        y_train, y_test = self.data_y[train_idx], self.data_y[test_idx]
        X_train_protected, X_test_protected = self.data_x_protected[train_idx], self.data_x_protected[test_idx]

        f_output_train = open("{}_{}_train.csv".format(self.plot_output, self.label_type), 'w')
        f_output_test = open("{}_{}_test.csv".format(self.plot_output, self.label_type), 'w')

        for eps in self.list_eps:
            res = red.expgrad(dataX=pd.DataFrame(np.delete(X_train, 0, 1)), dataA=X_train_protected, dataY=y_train,
                              # learner=LogisticRegression(),
                              learner=RandomForestClassifier(),
                              cons=moments.EO(), eps=eps)

            # Create the plots using unadjusted train and test data
            X_train_g0, X_test_g0, y_train_g0, y_test_g0, indices = self.split_train_test_data(self.data_x_g0, self.data_y_g0)
            X_train_g1, X_test_g1, y_train_g1, y_test_g1, indices = self.split_train_test_data(self.data_x_g1, self.data_y_g1)
            X_train_g01, X_test_g01, y_train_g01, y_test_g01, indices = self.split_train_test_data(
                np.array(self.data_x_g0 + self.data_x_g1),
                np.array(self.data_y_g0 + self.data_y_g1))

            # print(len(X_train_g01), len(X_train_g0), len(X_train_g1))
            X_train_a = self.data_x_protected[indices]
            weighted_preds_g0 = self.weighted_predictions(res, X_train_g0)
            weighted_preds_g1 = self.weighted_predictions(res, X_train_g1)
            weighted_preds_g01 = self.weighted_predictions(res, X_train_g01)
            #
            # tn0, fp0, fn0, tp0 = confusion_matrix(y_true=y_train_g0, y_pred=weighted_preds_g0, labels=[0, 1]).ravel()
            # # tn0, fp0, fn0, tp0 = confusion_matrix(y_true=y_train_g0, y_pred=weighted_preds_g0).ravel()
            # rate_fn_g0 = fn0 / (fn0 + tp0)
            # rate_fp_g0 = fp0 / (fp0 + tn0)
            # # print(tn0, fp0, fn0, tp0)
            #
            # tn1, fp1, fn1, tp1 = confusion_matrix(y_true=y_train_g1, y_pred=weighted_preds_g1, labels=[0, 1]).ravel()
            # # tn1, fp1, fn1, tp1 = confusion_matrix(y_true=y_train_g1, y_pred=weighted_preds_g1).ravel()
            #
            # rate_fn_g1 = fn1 / (fn1 + tp1)
            # rate_fp_g1 = fp1 / (fp1 + tn1)
            # # print(tn1, fp1, fn1, tp1)
            #
            # tn01, fp01, fn01, tp01 = confusion_matrix(y_true=y_train_g01, y_pred=weighted_preds_g01, labels=[0, 1]).ravel()
            # # tn01, fp01, fn01, tp01 = confusion_matrix(y_true=y_train_g01, y_pred=weighted_preds_g01).ravel()
            # error_train = (fp01 + fn01) / (fp01 + fn01 + tp01 + tn01)
            # disparity_train = abs(rate_fp_g1 - rate_fp_g0)
            #
            # print("{},{},{}".format(eps, disparity_train, error_train))
            # print("{},{},{}".format(eps, disparity_train, error_train), file=f_output_train)

            # if True:
            #     continue

            # print(len(X_train_a), len(y_train_g01), len(weighted_preds_g01))
            # print(np.count_nonzero(~np.isnan(X_train_a)))
            disparity_train = self.compute_FP(X_train_a.to_frame(), y_train_g01, weighted_preds_g01)
            error_train = self.compute_error(y_train_g01, weighted_preds_g01)
            # print(eps, fp)  # 0.00022960355120159193
            print("{},{},{}".format(eps, disparity_train, error_train))
            print("{},{},{}".format(eps, disparity_train, error_train), file=f_output_train)

            if True:
                continue

            # res = res._asdict()
            # TODO: compute false negative rates by individual points ...
            clf_cnt = 0
            classifiers, weights = res['classifiers'], res['weights'].tolist()
            error_train, rate_fn_g0, rate_fn_g1, rate_fp_g0, rate_fp_g1 = 0, 0, 0, 0, 0

            sum_fp_g0, sum_fp_g1 = 0, 0
            sum_fn = 0

            for idx in range(len(classifiers)):
                clf = classifiers[idx]
                w = weights[idx]
                clf_cnt += 1

                Y_pred = clf.predict(X_train_g0)
                tn0, fp0, fn0, tp0 = confusion_matrix(y_true=y_train_g0, y_pred=Y_pred, labels=[0, 1]).ravel()
                rate_fn_g0 += w * fn0 / (fn0 + tp0)
                rate_fp_g0 += w * fp0 / (fp0 + tn0)

                Y_pred = clf.predict(X_train_g1)
                tn1, fp1, fn1, tp1 = confusion_matrix(y_true=y_train_g1, y_pred=Y_pred, labels=[0, 1]).ravel()
                rate_fn_g1 += w * fn1 / (fn1 + tp1)
                rate_fp_g1 += w * fp1 / (fp1 + tn1)

                Y_pred = clf.predict(X_train_g01)
                tn01, fp01, fn01, tp01 = confusion_matrix(y_true=y_train_g01, y_pred=Y_pred, labels=[0, 1]).ravel()
                error_train += w * (fp01 + fn01) / (fp01 + fn01 + tp01 + tn01)

                # another way of computing FP rates

                Y_pred = clf.predict_proba(X_train_g0)
                for idx in range(len(Y_pred)):
                    sum_fp_g0 += Y_pred[idx][1] * w * (1 - y_train_g0[idx] - 1)
                    sum_fn += Y_pred[idx][0] * w

                Y_pred = clf.predict_proba(X_train_g1)
                for idx in range(len(Y_pred)):
                    sum_fp_g1 += Y_pred[idx][1] * w * (1 - y_train_g1[idx] - 1)
                    sum_fn += Y_pred[idx][0] * w

            if self.EO == 'FPR':
                disparity_train = abs(rate_fp_g0 - rate_fp_g1)
            elif self.EO == 'FNR':
                disparity_train = abs(rate_fn_g0 - rate_fn_g1)
            elif self.EO == 'BOTH':
                disparity_train = abs(rate_fp_g0 - rate_fp_g1) + abs(rate_fn_g0 - rate_fn_g1)

            # print("{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}".format(eps, rate_fp_g0, rate_fp_g1,
            #                                                                       rate_fn_g0, rate_fn_g1,
            #                                                                       disparity_train, error_train))
            # print("{},{},{}".format(eps, disparity_train, error_train), file=f_output_train)
            print("{},{},{}".format(eps, abs(sum_fp_g0 / len(X_train_g0) - sum_fp_g1 / len(X_train_g1)), error_train), file=f_output_train)
            # print(sum_fp_g0, sum_fp_g0 / len(Y_pred))
            # print(sum_fp_g1, sum_fp_g1 / len(Y_pred))
            # print(abs(sum_fp_g0 - sum_fp_g1))
            # print(sum_fn, sum_fn / len(Y_pred))

    @staticmethod
    def weighted_predictions(res_tuple, x):
        """
        Given res_tuple from expgrad, compute the weighted predictions
        over the dataset x
        """
        hs = res_tuple.classifiers
        weights = res_tuple.weights  # weights over classifiers
        preds = hs.apply(lambda h: h.predict(x))  # predictions
        return weights.dot(preds)

    @staticmethod
    def compute_error(y, weighted_pred):
        error = 0
        y = y.values
        # print(len(y), len(weighted_pred))
        for idx in range(len(weighted_pred)):
            if y[idx] == 1:
                error += 1 - weighted_pred[idx]
            else:
                error += weighted_pred[idx]
        return error / len(weighted_pred)

    @staticmethod
    def compute_FP(a, y, weighted_pred):
        disp = {}
        for a_val in [0, 1]:
            a_c = a
            # calculate Pr[ y-hat = 1 | y = 1 ]
            p_all = np.average(weighted_pred[y == 0])

            if len(weighted_pred[(y == 0) & (a_c == a_val)]) > 0:
                # calculate Pr[ y-hat = 1 | y = 1, a=1]
                p_sub = np.average(weighted_pred[(y == 0) & (a_c == a_val)])
                disp[(0, a_val)] = np.abs(p_all - p_sub)
        return max(disp.values())


    @staticmethod
    def compute_EO(a, y, weighted_pred):
        """
        Debug fn: compute equalized odds given weighted_pred
        assume a is already binarized
        """
        sens_attr = list(a.columns)
        disp = {}
        # for c in sens_attr:
        for y_val in [0, 1]:
            for a_val in [0, 1]:
                a_c = a
                # calculate Pr[ y-hat = 1 | y = 1 ]
                p_all = np.average(weighted_pred[y == y_val])

                if len(weighted_pred[(y == y_val) & (a_c == a_val)]) > 0:
                    # calculate Pr[ y-hat = 1 | y = 1, a=1]
                    p_sub = np.average(weighted_pred[(y == y_val) & (a_c == a_val)])
                    disp[(0, y_val, a_val)] = np.abs(p_all - p_sub)
        eps = max(disp.values())
        # (c_max, a_max, _) = max(disp, key=disp.get)
        # group_size = len(y[a[c_max] == a_max]) / len(y)
        return eps

    @staticmethod
    def compute_FN(a, y, weighted_pred):
        # sens_attr = list(a.columns)
        disp = {}
        # for c in sens_attr:
        for a_val in [0, 1]:
            # a_c = a[0]
            a_c = a
            # calculate Pr[ y-hat = 1 | y = 1 ]
            p_all = np.average(weighted_pred[y == 1])

            if len(weighted_pred[(y == 1) & (a_c == a_val)]) > 0:
                # calculate Pr[ y-hat = 1 | y = 1, a=1]
                p_sub = np.average(weighted_pred[(y == 1) & (a_c == a_val)])
                disp[(1, a_val)] = np.abs(p_all - p_sub)
        return max(disp.values())


    def plot_charts(self, filename=None):
        # TODO: check plotting correctness ...
        x = []
        y = []
        d = {}
        for line in open(filename, 'r'):
            model, unfairness, accuracy = line.strip().split(',')
            unfairness = float(unfairness)
            accuracy = float(accuracy)
            x.append(unfairness)
            y.append(accuracy)
            if unfairness in d:
                d[unfairness].append(accuracy)
            else:
                d[unfairness] = [accuracy]

        plt.xlabel('Disparity')
        plt.ylabel('Error Rate')
        # plt.xlabel('Recall')
        # plt.ylabel('Precision')
        # plt.plot(x, y, marker='o')
        # x, y = self.convex_env_train(x, y)
        plt.scatter(x, y, marker='o')
        plt.show()

    def plot_charts_quad(self, filename=None):
        # TODO: check plotting correctness ...
        x = []
        y = []
        d = {}
        dx = {}
        dy = {}
        filename = 'dataset/non_normalized_quad.csv'
        for line in open(filename, 'r'):
            lamb0, lamb1, unfairness, accuracy = line.strip().split(',')
            unfairness = float(unfairness)
            accuracy = float(accuracy)
            x.append(unfairness)
            y.append(accuracy)
            dx[str(lamb0) +str(lamb1)] = unfairness
            dy[str(lamb0) +str(lamb1)] = accuracy

            # dy[(lamb0, lamb1)] = accuracy

            # dx[lamb0] = unfairness
            # dy[lamb0] = accuracy

            if unfairness in d:
                d[unfairness].append(accuracy)
            else:
                d[unfairness] = [accuracy]

        plt.xlabel('Disparity')
        plt.ylabel('Error Rate')
        # plt.xlabel('Recall')
        # plt.ylabel('Precision')
        # plt.plot(x, y, marker='o')
        tuples = self.convex_env_train(dx, dy)
        return tuples
        # x, y = [], []
        # for tup in tuples:
        #     x.append(dx[tup])
        #     y.append(dy[tup])
        #
        #
        # # x, y = map(list, zip(*tuples))
        # # x = list(map(float, x))
        # # y = list(map(float, y))
        # plt.scatter(x, y, marker='o')
        # plt.show()

    def plot_charts_multiple(self, filename=None):
        # TODO: check plotting correctness ...
        x = []
        y = []
        d = {}
        filename = 'dataset/non_normalized3.csv'
        for line in open(filename, 'r'):
            model, unfairness, accuracy = line.strip().split(',')
            unfairness = float(unfairness)
            accuracy = float(accuracy)
            x.append(unfairness)
            y.append(accuracy)
            if unfairness in d:
                d[unfairness].append(accuracy)
            else:
                d[unfairness] = [accuracy]

        plt.xlabel('Disparity')
        plt.ylabel('Error Rate')
        # plt.xlabel('Recall')
        # plt.ylabel('Precision')
        # plt.plot(x, y, marker='o')
        # x, y = self.convex_env_train(x, y)
        plt.scatter(x, y, marker='o', label='non_normalized')

        x, y = [], []
        filename = 'dataset/normalized3.csv'
        for line in open(filename, 'r'):
            model, unfairness, accuracy = line.strip().split(',')
            unfairness = float(unfairness)
            accuracy = float(accuracy)
            x.append(unfairness)
            y.append(accuracy)
            if unfairness in d:
                d[unfairness].append(accuracy)
            else:
                d[unfairness] = [accuracy]

        # x, y = self.convex_env_train(x, y)
        plt.scatter(x, y, marker='^', label='normalized')

        x, y = [], []
        filename = 'dataset/plot_compas.csv'
        for line in open(filename, 'r'):
            model, unfairness, accuracy = line.strip().split(',')
            unfairness = float(unfairness)
            accuracy = float(accuracy)
            x.append(unfairness)
            y.append(accuracy)
            if unfairness in d:
                d[unfairness].append(accuracy)
            else:
                d[unfairness] = [accuracy]

        # x, y = self.convex_env_train(x, y)
        plt.scatter(x, y, marker='2', label='fairlearn')
        plt.legend(loc='best')

        plt.show()

    def generate_plot_data_lg(self):
        train, test, data, idx_X, idx_A, idx_y, data_name = Compas().create_data(a_type='race') # type doesn't matter..
        thresholds = np.linspace(0, 1, self.threshold_density, endpoint=True)

        fout = open('dataset/non_normalized_pareto_thre_t0.csv', 'w')
        tp = fp = fn = tn = 0
        tp_a0 = fp_a0 = fn_a0 = tn_a0 = 0

        clf = RandomForestClassifier()
        clf.fit(data[idx_X + idx_A], data[idx_y])

        # print(train[idx_X + idx_A])
        print(idx_X + idx_A)
        print(clf.feature_importances_)

        clf = LogisticRegression()
        clf.fit(data[idx_X + idx_A], data[idx_y])
        # y_pred = clf.predict(test[idx_X + idx_A])
        y_pred_prob = clf.predict_proba(data[idx_X + idx_A])

        for threshold in thresholds:
            tp = fp = fn = tn = 0
            for idx in range(len(y_pred_prob)):
                predicted_cls = 1 if y_pred_prob[idx][1] >= threshold else 0

                # sub_df.iloc[0]['A']
                if predicted_cls == 1 and data.iloc[idx][idx_y[0]] == 1:
                    tp += 1
                elif predicted_cls == 1 and data.iloc[idx][idx_y[0]] == 0:
                    fp += 1
                elif predicted_cls == 0 and data.iloc[idx][idx_y[0]] == 1:
                    fn += 1
                else:
                    tn += 1

            precision = 0 if tp + fp == 0 else tp / (tp + fp)
            recall = 0 if tp + fn == 0 else tp / (tp + fn)
            accuracy = (tp + tn) / (tp + tn + fp + fn)

            fpr = 0 if fp + tn == 0 else fp / (fp + tn)
            fnr = 0 if fn + tp == 0 else fn / (fn + tp)
            fp_a1, fn_a1, tp_a1, tn_a1 = 0, 0, 0, 0
            tp_a0, fp_a0, tn_a0, fn_a0 = tp, fp, tn, fn

            disp_fn, disp_fp = 0, 0
            lamb0, lamb1 = 0, 0
            att_type = 0
            # att_type = 1  # race
            # att_type = 2  # gender
            # print('disp_fn,disp_fp,threshold,precision,recall,fpr,fnr,tpa0,tpa1,fpa0,fpa1,fna0,fna1,tna0,fna1,type')
            # print('{:.5f},{:.5f},{:.3f},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{},{},{},{},{},{},{},{},{}'.format(disp_fn, disp_fp, threshold,
            #                                                                 precision, recall, fpr, fnr, accuracy,
            #                                                                 tp_a0, tp_a1, fp_a0, fp_a1,
            #                                                                 fn_a0, fn_a1, tn_a0, tn_a1, att_type),
            #       file=fout)

            # threshold,lamb0,lamb1,disp_fp,error,precision,recall,fpr,fnr,accuracy,tpa0,tpa1,fpa0,fpa1,fna0,fna1,tna0,fna1,type

            # threshold,lamb0,lamb1,disp_fp,precision,recall,fpr,fnr,accuracy,tpa0,tpa1,fpa0,fpa1,fna0,fna1,tna0,fna1,type
            print("{:.3f},{},{},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{},{},{},{},{},{},{},{},0".format(
                threshold, lamb0, lamb1, disp_fp, 1-accuracy,
                precision, recall,
                fpr, fnr, accuracy,
                tp_a0, tp_a1, fp_a0, fp_a1,
                fn_a0, fn_a1, tn_a0, tn_a1), file=fout)
            # print(
            #     "{},{},{:.3f},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{},{},{},{},{},{},{},{},0".format(
            #         lamb0, lamb1, threshold, disp_fn, disp_fp,
            #         precision, recall,
            #         fpr, fnr, accuracy,
            #         tp_a0, tp_a1, fp_a0, fp_a1,
            #         fn_a0, fn_a1, tn_a0, tn_a1), file=fout)

    def generate_plot_data(self):

        train, test, idx_X, idx_A, idx_y, data_name = Compas().create_data()
        thresholds = np.linspace(0, 1, self.threshold_density, endpoint=True)

        fout = open('data1.csv', 'w')

        for eps in self.list_eps:
            res = red.expgrad(dataX=train[idx_X],
                              dataA=train[idx_A].T.squeeze(),
                              dataY=train[idx_y].T.squeeze(),
                              learner=LogisticRegression(), cons=moments.EO(), eps=eps)

            weighted_preds = self.weighted_predictions(res, test[idx_X])
            # disparity_FP = self.compute_FP(test[idx_A].T.squeeze(), test[idx_y].T.squeeze(), weighted_preds)
            # disparity_FN = self.compute_FP(test[idx_A].T.squeeze(), test[idx_y].T.squeeze(), weighted_preds)

            for threshold in thresholds:
                print(eps, threshold)
                tp = fp = fn = tn = 0
                tp_a0 = fp_a0 = fn_a0 = tn_a0 = 0
                for idx in range(len(weighted_preds)):
                    predicted_cls = 1 if weighted_preds[idx] >= threshold else 0

                    # sub_df.iloc[0]['A']
                    if predicted_cls == 1 and test.iloc[idx][idx_y[0]] == 1:
                        case = 0  # tp
                        tp += 1
                        if test.iloc[idx][idx_A[0]] == 0:
                            tp_a0 += 1
                    elif predicted_cls == 1 and test.iloc[idx][idx_y[0]] == 0:
                        case = 1  # fp
                        fp += 1
                        if test.iloc[idx][idx_A[0]] == 0:
                            fp_a0 += 1
                    elif predicted_cls == 0 and test.iloc[idx][idx_y[0]] == 1:
                        case = 2  # fn
                        fn += 1
                        if test.iloc[idx][idx_A[0]] == 0:
                            fn_a0 += 1
                    else:
                        case = 3  # tn
                        tn += 1
                        if test.iloc[idx][idx_A[0]] == 0:
                            tn_a0 += 1

                precision = 0 if tp + fp == 0 else tp / (tp + fp)
                recall = 0 if tp + fn == 0 else tp / (tp + fn)
                accuracy = (tp + tn) / (tp + tn + fp + fn)

                fpr = 0 if fp + tn == 0 else fp / (fp + tn)
                fnr = 0 if fn + tp == 0 else fn / (fn + tp)

                fp_a1, fn_a1, tp_a1, tn_a1 = fp - fp_a0, fn - fn_a0, tp - tp_a0, tn - tn_a0
                rate_fp_a0 = 0 if fp_a0 + tn_a0 == 0 else fp_a0 / (fp_a0 + tn_a0)
                rate_fn_a0 = 0 if fn_a0 + tp_a0 == 0 else fn_a0 / (fn_a0 + tp_a0)
                rate_fp_a1 = 0 if fp_a1 + tn_a1 == 0 else fp_a1 / (fp_a1 + tn_a1)
                rate_fn_a1 = 0 if fn_a1 + tp_a1 == 0 else fn_a1 / (fn_a1 + tp_a1)

                disp_fn, disp_fp = abs(rate_fn_a0 - rate_fn_a1), abs(rate_fp_a0 - rate_fp_a1)
                att_type = 1  # race
                # att_type = 2  # gender
                # print('disp_fn,disp_fp,threshold,precision,recall,fpr,fnr,tpa0,tpa1,fpa0,fpa1,fna0,fna1,tna0,fna1,type')
                print('{:.5f},{:.5f},{:.3f},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{},{},{},{},{},{},{},{},{}'.format(disp_fn, disp_fp, threshold,
                                                                                precision, recall, fpr, fnr, accuracy,
                                                                                tp_a0, tp_a1, fp_a0, fp_a1,
                                                                                fn_a0, fn_a1, tn_a0, tn_a1, att_type),
                                                                                file=fout)




            # for threshold in thresholds:
            #     predicted_cls = 1 if y_score[1] >= threshold else 0
            #
            # disparity_train = self.compute_FP(test[idx_A].T.squeeze(), test[idx_y].T.squeeze(),
            #                                   weighted_preds)
            # error_train = self.compute_error(test[idx_y].T.squeeze(), weighted_preds)
            #
            # print("{},{},{}".format(eps, disparity_train, error_train))
            # print("{},{},{}".format(eps, disparity_train, error_train), file=f_output)


    def analysis_on_compas_data(self):
        # distribution of race and label

        # precision-recall trade off
        train, test, idx_X, idx_A, idx_y, data_name = Compas().create_data()
        print(len(idx_X), len(idx_A), len(idx_y))
        data = pd.concat([train, test])
        print(train.shape, test.shape, data.shape)

        y0 = data[data[idx_y[0]] == 0]  # 5099 is_not_recid
        y1 = data[data[idx_y[0]] == 1]  # 2819 is_recid
        print(y0.shape, y1.shape)

        # white 1, black 0
        a0 = data[data[idx_A[0]] == 0]  # 4671 black
        a1 = data[data[idx_A[0]] == 1]  # 3247 white
        print(a0.shape, a1.shape)
        # female: 1675, male: 6243

        clf = LogisticRegression()  # default P>0.5
        # clf = RandomForestClassifier()
        clf.fit(train[idx_X+idx_A], train[idx_y])

        y_pred = clf.predict(test[idx_X+idx_A])
        y_pred_prob = clf.predict_proba(test[idx_X+idx_A])

        print(y_pred.shape)
        tn, fp, fn, tp = confusion_matrix(test[idx_y[0]], y_pred, [0, 1]).ravel()
        accuracy = (tp + tn) / (tn + fp + fn + tp)
        print(accuracy)

        thresholds = np.linspace(0, 1, self.threshold_density, endpoint=True)
        filename = 'compas.csv'
        f_output = open(filename, 'w')
        for threshold in thresholds:
            list_Y_pred = []

            for score in y_pred_prob:
                list_Y_pred.append(1 if score[1] >= threshold else 0)

            threshold = str(round(threshold, self.decimal))

            # Predicting damaging edits
            tn, fp, fn, tp = confusion_matrix(y_true=test[idx_y[0]], y_pred=list_Y_pred).ravel()
            rate_fp = fp / (fp + tn)
            rate_tp = tp / (tp + fn)  # sensitivity/recall/true positive rates
            rate_fn = fn / (fn + tp)
            rate_tn = tn / (tn + fp)  # specificity/true negative rates
            precision = 0 if tp + fp == 0 else tp / (tp + fp)
            recall = 0 if tp + fn == 0 else tp / (tp + fn)

            # print(rate_fn, rate_fp)

            # print("{},{},{}".format(threshold, recall, precision), file=f_output)
            print("{},{},{}".format(threshold, rate_fp, rate_fn), file=f_output)
            # print("{},{},{}".format(threshold, recall, precision))

        f_output.close()
        self.plot_charts(filename)

    def learner_normalized(self, x, a, y, learner, lamb0, lamb1):

        ones = np.ones(y.shape)
        p0, p1 = len(ones[(a == 1) & (y == 0)]) / len(ones), len(ones[(a == 1) & (y == 1)]) / len(ones)
        p00, p11 = len(y[y == 0]) / len(y), len(y[y == 1]) / len(y)

        vec0, vec1 = ((y == 0) & (a == 1)).astype(int), ((y == 1) & (a == 1)).astype(int)
        vec00, vec11 = (y == 0).astype(int), (y == 1).astype(int)

        cost1 = (1 - y) + lamb0 * (vec0 / p0 - vec00 / p00)
        cost0 = y + lamb1 * (vec1 / p1 - vec11 / p11)
        W = abs(cost0 - cost1)
        Y = 1 * (cost0 > cost1)

        learner.fit(x, Y, W)
        return learner

    def learner_non_normalized(self, x, a, y, learner, lamb0, lamb1):
        vec0, vec1 = ((y == 0) & (a == 1)).astype(int), ((y == 1) & (a == 1)).astype(int)
        vec00, vec11 = (y == 0).astype(int), (y == 1).astype(int)

        cost1 = (1 - y) + lamb1 * (vec0 - vec00)
        cost0 = y + lamb0 * (vec1 - vec11)
        W = abs(cost0 - cost1)
        Y = 1 * (cost0 > cost1)

        learner.fit(x, Y, W)
        return learner

    def lambs_tri(self):
        train, test, data, idx_X, idx_A, idx_y, data_name = Compas().create_data()
        # type2 sex
        filename = 'dataset/non_normalized_pareto_t1.csv'
        f_output = open(filename, 'w')
        x, a, y = data[idx_X], data[idx_A[0]], data[idx_y[0]]

        # read the points on the pareto curve
        tuples = self.plot_charts_quad()

        for lamb0 in self.list_eps:
            for lamb1 in self.list_eps:
                tup = str(lamb0)+str(lamb1)
                if tup not in tuples:
                    continue

                clf = self.learner_non_normalized(x, a, y, LogisticRegression(), lamb0, lamb1)
                # y_pred = clf.predict(data[idx_X])
                pred_prob_y = clf.predict_proba(data[idx_X])[:, 1]

                # thresholds = np.linspace(0, 1, self.threshold_density, endpoint=True)
                thresholds = [0.5]
                for threshold in thresholds:
                    y_pred = np.where(pred_prob_y >= threshold, 1, 0)
                    tn, fp, fn, tp = confusion_matrix(y, y_pred, [0, 1]).ravel()

                    # disparity_train = self.compute_FP(data[idx_A].T.squeeze(), data[idx_y].T.squeeze(), pred_prob_y)
                    # error_train = self.compute_error(data[idx_y].T.squeeze(), pred_prob_y)
                    error_train = sum(np.abs(y - y_pred)) / len(y)

                    # disparity for the two groups
                    fpr, fnr = fp / (fp+tn), fn / (fn + tp)
                    precision = 0 if tp + fp == 0 else tp / (tp + fp)
                    recall = 0 if tp + fn == 0 else tp / (tp + fn)
                    accuracy = (tp + tn) / (tp + tn + fp + fn)

                    y0, y1 = y[a == 0], y[a == 1]
                    y0_pred, y1_pred = y_pred[a == 0], y_pred[a == 1]

                    tn0, fp0, fn0, tp0 = confusion_matrix(y0, y0_pred, [0, 1]).ravel()
                    tn1, fp1, fn1, tp1 = confusion_matrix(y1, y1_pred, [0, 1]).ravel()
                    fpr0, fpr1 = fp0 / (fp0 + tn0), fp1 / (fp1 + tn1)
                    fnr0, fnr1 = fn0 / (fn0 + tp0), fn1 / (fn1 + tp1)

                    disparity_fpr = max(abs(fpr-fpr0), abs(fpr-fpr1))
                    disparity_fnr = max(abs(fnr - fnr0), abs(fnr - fnr1))

                    # disp_fn,disp_fp,threshold,precision,recall,fpr,fnr,accuracy,tpa0,tpa1,fpa0,fpa1,fna0,fna1,tna0,fna1,type
                    print("{:.5f},{:.5f},{:.3f},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{},{},{},{},{},{},{},{},1".format(
                        disparity_fnr, disparity_fpr, threshold,
                        precision, recall,
                        fpr, fnr, accuracy,
                        tp0, tp1, fp0, fp1,
                        fn0, fn1, tn0, tn1), file=f_output)

                    # lamb0,lamb1,threshold,disp_fn,disp_fp,precision,recall,fpr,fnr,accuracy,tpa0,tpa1,fpa0,fpa1,fna0,fna1,tna0,fna1,type
                    # print(
                    #     "{},{},{:.3f},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{},{},{},{},{},{},{},{},1".format(
                    #         lamb0, lamb1, threshold, disparity_fnr, disparity_fpr,
                    #         precision, recall,
                    #         fpr, fnr, accuracy,
                    #         tp0, tp1, fp0, fp1,
                    #         fn0, fn1, tn0, tn1), file=f_output)

                    print("{},{},{:.3f},{:.5f},{:.5f}".format(lamb0, lamb1, threshold, disparity_fpr, error_train))
                    # print("{},{},{}".format(lamb0, disparity_fpr, error_train), file=f_output)

        f_output.close()
        # self.plot_charts(filename)

    def learner_non_normalized_abs(self, x, a, y, learner, lamb0, lamb1):
        vec0, vec00 = ((y == 0) & (a == 1)).astype(int), ((y == 0) & (a == 0)).astype(int)
        vec1, vec11 = ((y == 1) & (a == 1)).astype(int), ((y == 1) & (a == 0)).astype(int)

        cost1 = (1 - y) + lamb1 * (vec0 - vec00)
        cost0 = y + lamb0 * (vec1 - vec11)
        W = abs(cost0 - cost1)
        Y = 1 * (cost0 > cost1)

        learner.fit(x, Y, W)
        return learner

    def lambs_abs(self):

        a_type = 'race'
        train, test, data, idx_X, idx_A, idx_y, data_name = Compas().create_data(a_type)
        # a_type: 1 race, 2, gender
        a_type = 1 if a_type == 'race' else 2

        filename = 'dataset/non_normalized_quad{}.csv'.format(a_type)
        f_output = open(filename, 'w')
        x, a, y = data[idx_X], data[idx_A[0]], data[idx_y[0]]

        # read the points on the pareto curve
        tuples = self.plot_charts_quad()

        range_lamb0 = self.list_eps  # for fp
        range_lamb1 = [0]  # for fn
        for lamb0 in range_lamb0:
            for lamb1 in range_lamb1:

                tup = str(lamb0)+str(lamb1)
                if tup not in tuples:
                    continue

                clf = self.learner_non_normalized_abs(x, a, y, LogisticRegression(), lamb0, lamb1)
                # y_pred = clf.predict(data[idx_X])
                pred_prob_y = clf.predict_proba(data[idx_X])[:, 1]

                # thresholds = np.linspace(0, 1, self.threshold_density, endpoint=True)
                thresholds = [0.5]
                for threshold in thresholds:
                    y_pred = np.where(pred_prob_y >= threshold, 1, 0)
                    tn, fp, fn, tp = confusion_matrix(y, y_pred, [0, 1]).ravel()

                    # disparity_train = self.compute_FP(data[idx_A].T.squeeze(), data[idx_y].T.squeeze(), pred_prob_y)
                    # error_train = self.compute_error(data[idx_y].T.squeeze(), pred_prob_y)
                    error_train = sum(np.abs(y - y_pred)) / len(y)

                    # disparity for the two groups
                    fpr, fnr = fp / (fp+tn), fn / (fn + tp)
                    precision = 0 if tp + fp == 0 else tp / (tp + fp)
                    recall = 0 if tp + fn == 0 else tp / (tp + fn)
                    accuracy = (tp + tn) / (tp + tn + fp + fn)

                    y0, y1 = y[a == 0], y[a == 1]
                    y0_pred, y1_pred = y_pred[a == 0], y_pred[a == 1]

                    tn0, fp0, fn0, tp0 = confusion_matrix(y0, y0_pred, [0, 1]).ravel()
                    tn1, fp1, fn1, tp1 = confusion_matrix(y1, y1_pred, [0, 1]).ravel()
                    fpr0, fpr1 = fp0 / (fp0 + tn0), fp1 / (fp1 + tn1)
                    fnr0, fnr1 = fn0 / (fn0 + tp0), fn1 / (fn1 + tp1)

                    # disparity for both fn and fp
                    disparity = max(abs(fp0 - fp1) / 2, abs(fn0 - fn1) / 2)

                    # disparity for only fp
                    disparity = abs(fp0 - fp1) / 2
                    # disparity_fpr = abs(fp0 - fp1) / 2
                    # disparity_fpr = abs(fn0 - fn1) / 2
                    # disparity_fpr = max(abs(fpr-fpr0), abs(fpr-fpr1))
                    # disparity_fnr = max(abs(fnr - fnr0), abs(fnr - fnr1))

                    # disp_fn,disp_fp,threshold,precision,recall,fpr,fnr,accuracy,tpa0,tpa1,fpa0,fpa1,fna0,fna1,tna0,fna1,type
                    # print("{},{:.5f},{:.3f},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{},{},{},{},{},{},{},{},{}".format(
                    #     lamb0, disparity_fpr, threshold,
                    #     precision, recall,
                    #     fpr, fnr, accuracy,
                    #     tp0, tp1, fp0, fp1,
                    #     fn0, fn1, tn0, tn1, a_type), file=f_output)

                    # lamb0,lamb1,threshold,disp_fn,disp_fp,precision,recall,fpr,fnr,accuracy,tpa0,tpa1,fpa0,fpa1,fna0,fna1,tna0,fna1,type
                    # print(
                    #     "{},{},{:.3f},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{},{},{},{},{},{},{},{},{}".format(
                    #         lamb0, lamb1, threshold, disparity_fnr, disparity_fpr,
                    #         precision, recall,
                    #         fpr, fnr, accuracy,
                    #         tp0, tp1, fp0, fp1,
                    #         fn0, fn1, tn0, tn1, a_type), file=f_output)

                    print("{},{},{:.3f},{:.5f},{:.5f}".format(lamb0, lamb1, threshold, disparity, error_train))
                    # print("{},{},{},{}".format(lamb0, lamb1, disparity, error_train), file=f_output)
                    print("{},{},{}".format(lamb0, disparity, error_train), file=f_output)

        f_output.close()
        self.plot_charts(filename)

    def lambs_abs_pareto(self):

        a_type = 'sex'
        train, test, data, idx_X, idx_A, idx_y, data_name = Compas().create_data(a_type)
        # a_type: 1 race, 2, gender
        a_type = 1 if a_type == 'race' else 2

        filename = 'dataset/non_normalized_pareto_thred_t{}.csv'.format(a_type)
        f_output = open(filename, 'w')
        x, a, y = data[idx_X], data[idx_A[0]], data[idx_y[0]]

        # read the points on the pareto curve
        tuples = self.plot_charts_quad()
        # creating the entire curve ...
        range_lamb0 = self.list_eps  # for fp
        range_lamb1 = [0]  # for fn

        dX_vals = {}
        d_keys_vals = {}
        thresholds = np.linspace(0, 1, self.threshold_density, endpoint=True)
        for threshold in thresholds:
            Xs = {}
            Ys = {}
            d_pairs_vals = {}
            for lamb0 in range_lamb0:
                for lamb1 in range_lamb1:
                    clf = self.learner_non_normalized_abs(x, a, y, LogisticRegression(), lamb0, lamb1)
                    pred_prob_y = clf.predict_proba(data[idx_X])[:, 1]

                    y_pred = np.where(pred_prob_y >= threshold, 1, 0)
                    tn, fp, fn, tp = confusion_matrix(y, y_pred, [0, 1]).ravel()

                    error = sum(np.abs(y - y_pred)) / len(y)

                    # disparity for the two groups
                    fpr, fnr = fp / (fp + tn), fn / (fn + tp)
                    precision = 0 if tp + fp == 0 else tp / (tp + fp)
                    recall = 0 if tp + fn == 0 else tp / (tp + fn)
                    accuracy = (tp + tn) / (tp + tn + fp + fn)

                    y0, y1 = y[a == 0], y[a == 1]
                    y0_pred, y1_pred = y_pred[a == 0], y_pred[a == 1]

                    tn0, fp0, fn0, tp0 = confusion_matrix(y0, y0_pred, [0, 1]).ravel()
                    tn1, fp1, fn1, tp1 = confusion_matrix(y1, y1_pred, [0, 1]).ravel()
                    fpr0, fpr1 = fp0 / (fp0 + tn0), fp1 / (fp1 + tn1)
                    fnr0, fnr1 = fn0 / (fn0 + tp0), fn1 / (fn1 + tp1)

                    # disparity for only fp
                    disparity = abs(fp0 - fp1) / 2

                    # print("{},{},{},{}".format(lamb0, lamb1, disparity, error), file=f_output)

                    Xs[str(lamb0) + str(lamb1)] = disparity
                    Ys[str(lamb0) + str(lamb1)] = error
                    print("{},{},{},{},{}".format(threshold, lamb0, lamb1, disparity, error))
                    # threshold,lamb0,lamb1,disp_fp,precision,recall,fpr,fnr,accuracy,tpa0,tpa1,fpa0,fpa1,fna0,fna1,tna0,fna1,type
                    vals = "{:.3f},{:.3f},{:.3f},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{},{},{},{},{},{},{},{},{}".format(
                            threshold, lamb0, lamb1, disparity, error,
                            precision, recall,
                            fpr, fnr, accuracy,
                            tp0, tp1, fp0, fp1,
                            fn0, fn1, tn0, tn1, a_type)
                    d_pairs_vals[str(lamb0)+str(lamb1)] = vals
            # compute pareto curve under this threshold
            tuples = self.convex_env_train(Xs, Ys)

            # only select data points for this threshold
            d_sub_pairs_vals = {}
            for tup in tuples:
                vals = d_pairs_vals[tup]
                d_sub_pairs_vals[tup] = vals
            d_keys_vals[threshold] = d_sub_pairs_vals

        # rescan to print in order of threshold and lamb0
        for threshold in thresholds:
            for lamb0 in range_lamb0:
                for lamb1 in range_lamb1:
                    d_sub_pairs_vals = d_keys_vals[threshold]
                    tup = str(lamb0) + str(lamb1)
                    if tup in d_sub_pairs_vals:
                        print(d_sub_pairs_vals[tup], file=f_output)
        f_output.close()

    def lambs(self):
        train, test, data, idx_X, idx_A, idx_y, data_name = Compas().create_data()
        filename = 'dataset/non_normalized_quad.csv'
        f_output = open(filename, 'w')
        x, a, y = data[idx_X], data[idx_A[0]], data[idx_y[0]]
        for lamb0 in self.list_eps:
            for lamb1 in self.list_eps:
                clf = self.learner_non_normalized(x, a, y, LogisticRegression(), lamb0, lamb1)
                y_pred = clf.predict(data[idx_X])
                pred_prob_y = clf.predict_proba(data[idx_X])[:, 1]

                tn, fp, fn, tp = confusion_matrix(y, y_pred, [0, 1]).ravel()

                # disparity_train = self.compute_FP(data[idx_A].T.squeeze(), data[idx_y].T.squeeze(), pred_prob_y)
                # error_train = self.compute_error(data[idx_y].T.squeeze(), pred_prob_y)
                error_train = sum(np.abs(y - y_pred)) / len(y)

                # disparity for the two groups
                fpr, fnr = fp / (fp+tn), fn / (fn + tp)
                precision = 0 if tp + fp == 0 else tp / (tp + fp)
                recall = 0 if tp + fn == 0 else tp / (tp + fn)
                accuracy = (tp + tn) / (tp + tn + fp + fn)

                y0, y1 = y[a == 0], y[a == 1]
                y0_pred, y1_pred = y_pred[a == 0], y_pred[a == 1]

                tn0, fp0, fn0, tp0 = confusion_matrix(y0, y0_pred, [0, 1]).ravel()
                tn1, fp1, fn1, tp1 = confusion_matrix(y1, y1_pred, [0, 1]).ravel()
                fpr0, fpr1 = fp0 / (fp0 + tn0), fp1 / (fp1 + tn1)
                fnr0, fnr1 = fn0 / (fn0 + tp0), fn1 / (fn1 + tp1)

                disparity_fpr = max(abs(fpr-fpr0), abs(fpr-fpr1))
                disparity_fnr = max(abs(fnr - fnr0), abs(fnr - fnr1))

                # print("{},{},{:.3f},{:.5f},{:.5f}".format(lamb0, lamb1, threshold, disparity_fpr, error_train))
                print("{},{},{},{}".format(lamb0, lamb1, disparity_fpr, error_train))
                print("{},{},{},{}".format(lamb0, lamb1, disparity_fpr, error_train), file=f_output)

        f_output.close()
        self.plot_charts_quad(filename)

    def convex_env_train(self, Xs, Ys):
        """
        Identify the convex envelope on the set of models
        from the train set.
        """
        # Sort the list in either ascending or descending order of the
        # items values in Xs
        key_X_pairs = sorted(Xs.items(), key=lambda x: x[1],
                             reverse=False)  # this is a list of (key, val) pairs
        # Start the Pareto frontier with the first key value in the sorted list
        p_front = [key_X_pairs[0][0]]
        # Loop through the sorted list
        count = 0
        for (key, X) in key_X_pairs:
            if Ys[key] <= Ys[p_front[-1]]:  # Look for lower values of Y
                if count > 0:
                    p_front.append(key)
            count = count + 1
        return self.remove_interior(p_front, Xs, Ys)

    def remove_interior(self, p_front, Xs, Ys):
        if len(p_front) < 3:
            return p_front
        [k1, k2, k3] = p_front[:3]
        x1 = Xs[k1]
        y1 = Ys[k1]
        x2 = Xs[k2]
        y2 = Ys[k2]
        x3 = Xs[k3]
        y3 = Ys[k3]
        # compute the linear interpolation between 1 and 3 when x = x2
        if x1 == x3:  # equal values
            return self.remove_interior([k1] + p_front[3:], Xs, Ys)
        else:
            alpha = (x2 - x1) / (x3 - x1)
            y_hat = y1 - (y1 - y3) * alpha
            if y_hat >= y2:  # keep the triplet
                return [k1] + self.remove_interior(p_front[1:], Xs, Ys)
            else:  # remove 2
                return self.remove_interior([k1, k3] + p_front[3:], Xs, Ys)

    def replicate_results(self):

        # Adult(), Compas(), Dutch(), Lawschool(), ParserWiki()
        for obj in [ParserWiki()]:

            train, test, data, idx_X, idx_A, idx_y, data_name = obj.create_data()
            # print(data_name, train.shape, test.shape, len(idx_X), len(idx_A), len(idx_y))
            # print(train[idx_X].shape, train[idx_A].shape, train[idx_y].shape)

            clf = RandomForestClassifier()
            clf.fit(train[idx_X], train[idx_y])
            pred = clf.predict(test[idx_X])
            tn0, fp0, fn0, tp0 = confusion_matrix(test[idx_y], pred).ravel()
            # tn0, fp0, fn0, tp0 = confusion_matrix(y0, yp_0, [0, 1]).ravel()
            # print(tn0, fp0, fn0, tp0)  # 1494 4 1 379  # 9301 32 298 75

            train_full = data
            # To equalize FP rate: make all the positive examples (y=1) belong to the same group (a = 1)
            # train_adjusted = train.drop(train[(train.gender == 0) & (train.label == 1)].index)
            # train.loc[train[idx_y[0]] == 1, idx_A[0]] = 0
            # train.loc[train[idx_y[0]] == 1, idx_A[0]] = 1

            # y1 = train[idx_y[0]] == 1
            # a1 = train[idx_A[0]] == 1
            # print(train[y1 & a1].shape)
            #
            # y0 = train[idx_y[0]] == 0
            # a1 = train[idx_A[0]] == 1
            # print(train[y0 & a1].shape)
            #
            # y1 = train[idx_y[0]] == 1
            # a0 = train[idx_A[0]] == 0
            # print(train[y1 & a0].shape)
            #
            # y0 = train[idx_y[0]] == 0
            # a1 = train[idx_A[0]] == 1
            # print(train[y0 & a1].shape)

            a0 = data[data[idx_A[1]] == 0]
            a1 = data[data[idx_A[1]] == 1]
            print(a0.shape, a1.shape)  # 455 v.s. 296

            y0 = train[train[idx_y[0]] == 0]
            y1 = train[train[idx_y[0]] == 1]
            print(y0.shape, y1.shape)  # 359 v.s. 392

            train_adjusted = data

            filename = 'dataset/plot_{}.csv'.format(data_name)
            print(filename)
            f_output = open(filename, 'w')
            for eps in self.list_eps:
                x, a, y = data[idx_X], data[idx_A[0]], data[idx_y[0]]

                # res = red.expgrad(dataX=train_adjusted[idx_X],
                #                   dataA=train_adjusted[idx_A].T.squeeze(),
                #                   dataY=train_adjusted[idx_y].T.squeeze(),
                #                   learner=LogisticRegression(), cons=moments.EO(), eps=eps)
                res = red.expgrad(x, a, y,
                                  learner=LogisticRegression(), cons=moments.EO(), eps=eps)
                                  # learner=GradientBoostingClassifier(learning_rate=0.01, max_depth=7, max_features="sqrt",
                                  #                            n_estimators=700),
                                  # learner=RandomForestClassifier(),
                                  # cons=moments.EO(), eps=eps)

                weighted_preds = self.weighted_predictions(res, train_full[idx_X])

                # print(type(weighted_preds))
                # y_pred = np.where(weighted_preds > 0.5, 1, 0)
                # tn, fp, fn, tp = confusion_matrix(y, y_pred, [0, 1]).ravel()

                ########################################################################
                # # weighted_preds = int(weighted_preds)  # to int ...
                # print(type(weighted_preds))
                # weighted_preds = weighted_preds.astype('int32')
                # yp_0 = weighted_preds[train_full[idx_A[0]] == 0]
                # yp_1 = weighted_preds[train_full[idx_A[0]] == 1]
                #
                # y0 = train_full[idx_y][train_full[idx_A[0]] == 0]
                # y1 = train_full[idx_y][train_full[idx_A[0]] == 1]
                #
                # print(yp_0.shape, yp_1.shape, y0.shape, y1.shape)
                #
                # tn0, fp0, fn0, tp0 = confusion_matrix(y0, yp_0, [0, 1]).ravel()
                # tn1, fp1, fn1, tp1 = confusion_matrix(y1, yp_1, [0, 1]).ravel()
                # print(tn0, fp0, fn0, tp0)
                # print(tn1, fp1, fn1, tp1)
                # disparity_fpr = abs(fp0 / (fp0 + tn0) - fp1 / (fp1 + tn1))
                # disparity_fnr = abs(fn0 / (fn0 + tp0) - fn1 / (fn1 + tp1))
                # error = (fp0 + fn0 + fp1 + fn1) / (tn0 + fp0 + fn0 + tp0 + tn1 + fp1 + fn1 + tp1)
                # print(eps, disparity_fpr, disparity_fnr, error)

                # compute the values ...

                ########################################################################

                disparity_train = self.compute_FP(a, y, weighted_preds)
                # disparity_train = self.compute_FN(train_full[idx_A].T.squeeze(), train_full[idx_y].T.squeeze(),
                #                                   weighted_preds)
                error_train = self.compute_error(y, weighted_preds)

                print("{},{},{}".format(eps, disparity_train, error_train))
                print("{},{},{}".format(eps, disparity_train, error_train), file=f_output)

                ###########################################################################
                # # binary prediction
                # error_train = sum(np.abs(y - y_pred)) / len(y)
                #
                # # disparity for the two groups
                # fpr = fp / (fp + tn)
                #
                # y0, y1 = y[a == 0], y[a == 1]
                # y0_pred, y1_pred = y_pred[a == 0], y_pred[a == 1]
                #
                # tn0, fp0, fn0, tp0 = confusion_matrix(y0, y0_pred, [0, 1]).ravel()
                # tn1, fp1, fn1, tp1 = confusion_matrix(y1, y1_pred, [0, 1]).ravel()
                # fpr0 = fp0 / (fp0 + tn0)
                # fpr1 = fp1 / (fp1 + tn1)
                #
                # disparity_train = max(abs(fpr - fpr0), abs(fpr - fpr1))
                # print("{},{},{}".format(eps, disparity_train, error_train))
                # print("{},{},{}".format(eps, disparity_train, error_train), file=f_output)

            f_output.close()
            self.plot_charts(filename)


def main():
    runner = PredictionFairness(sys.argv[1])
    # runner.load_data()
    # runner.data_reformulation()
    # runner.run_cross_validation()
    # runner.run_train_test_split_fairlearn()
    # runner.run_train_test_split_baseline2()
    # runner.plot_charts()

    # runner.replicate_results()
    # runner.lambs()
    # runner.plot_charts_quad()
    # runner.lambs_tri()
    # runner.lambs_abs()
    # runner.lambs_abs_pareto()

    # runner.plot_charts_multiple()
    # runner.analysis_on_compas_data()
    # runner.generate_plot_data()
    runner.generate_plot_data_lg()

if __name__ == '__main__':
    main()
