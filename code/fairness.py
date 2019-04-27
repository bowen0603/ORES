from parser_adult import Adult
from parser_dutch import Dutch
from parser_lawschool import Lawschool
from parser_campus import Campus
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

__author__ = 'bobo'


class PredictionFairness:

    def __init__(self, EO):
        self.decimal = 3
        self.n_folds = 1
        self.N = 19412
        self.eps = 0.100
        self.threshold_density = 101
        self.list_eps = [0.001, 0.025, 0.005, 0.0075, 0.01, 0.0125, 0.015, 0.0175, 0.02, 0.025, 0.03, 0.04,
                         0.05, 0.075, 0.1, 0.2, 0.3, 0.4, 0.5]
        # self.list_eps = [0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.3, 0.5]
        # self.list_eps = [0.001, 0.01, 0.03, 0.05, 0.075, 0.1, 0.3, 0.5]

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
        # Adult(), Campus(), Dutch(), Lawschool(), ParserWiki()
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
        # return weighted predictions
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
        # sens_attr = list(a.columns)
        disp = {}
        # for c in sens_attr:
        for a_val in [0, 1]:
            # a_c = a[0]
            a_c = a
            # calculate Pr[ y-hat = 1 | y = 1 ]
            p_all = np.average(weighted_pred[y == 0])

            if len(weighted_pred[(y == 0) & (a_c == a_val)]) > 0:
                # calculate Pr[ y-hat = 1 | y = 1, a=1]
                p_sub = np.average(weighted_pred[(y == 0) & (a_c == a_val)])
                disp[(0, a_val)] = np.abs(p_all - p_sub)
        return max(disp.values())

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
                disp[(0, a_val)] = np.abs(p_all - p_sub)
        return max(disp.values())

    def plot_charts(self, filename=None):
        # TODO: check plotting correctness ...
        x = []
        y = []
        d = {}
        # filename='dataset/plot_lawschool.csv'
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
        # TODO: remove points whose disparity is greater than 0.5 to see details
        # for key, val in d.items():
        #     d[key] = sum(d[key]) / len(d[key])
        #
        # for unfairness, accuracy in sorted(d.items(), key=operator.itemgetter(0)):  # sorted by unfairness
        #     # y.append(unfairness)
        #     # x.append(accuracy)
        #     x.append(unfairness)
        #     y.append(accuracy)

        # if self.label_type == 'quality':
        #     # equalizing FN
        #     plt.xlabel('Unfairness (Disparity of False Negative Rates/Quality Control)')
        #     plt.ylabel('Prediction accuracy')
        #     plt.title('Value Trade-off between Unfairness and Prediction Accuracy\n(Editing Quality)')
        # elif self.label_type == 'intention':
        #     # equalizing FN
        #     plt.ylabel('Unfairness (Disparity of False Negative Rates/Motivation Protection)')
        #     plt.xlabel('Prediction accuracy')
        #     plt.title('Value Trade-off between Unfairness and Prediction Accuracy\n(Editing Intention)')
        # else:
        #     # TODO: equalizing FP rates
        #     print("Invalid prediction label ..")
        #     return

        plt.xlabel('Disparity')
        plt.ylabel('Error Rate')
        # plt.plot(x, y, marker='o')
        plt.scatter(x, y, marker='o')
        plt.show()

    def replicate_results(self):

        # Adult(), Campus(), Dutch(), Lawschool(), ParserWiki()
        for obj in [ParserWiki()]:

            train, test, idx_X, idx_A, idx_y, data_name = obj.create_data()
            print(data_name, train.shape, test.shape, len(idx_X), len(idx_A), len(idx_y))

            train_full = test
            # To equalize FP rate: make all the positive examples (y=1) belong to the same group (a = 1)
            # train_adjusted = train.drop(train[(train.gender == 0) & (train.label == 1)].index)
            # train.loc[train[idx_y[0]] == 1, idx_A[0]] = 0
            train_adjusted = train

            filename = 'dataset/plot_{}.csv'.format(data_name)
            print(filename)
            f_output = open(filename, 'w')
            for eps in self.list_eps:
                res = red.expgrad(dataX=train_adjusted[idx_X],
                                  dataA=train_adjusted[idx_A].T.squeeze(),
                                  dataY=train_adjusted[idx_y].T.squeeze(),
                                  # learner=LogisticRegression(), cons=moments.EO(), eps=eps)
                                  learner=RandomForestClassifier(), cons=moments.EO(), eps=eps)

                weighted_preds = self.weighted_predictions(res, train_full[idx_X])
                # disparity_train = self.compute_FP(train_full[idx_A].T.squeeze(), train_full[idx_y].T.squeeze(),
                #                                   weighted_preds)
                disparity_train = self.compute_FN(train_full[idx_A].T.squeeze(), train_full[idx_y].T.squeeze(),
                                                  weighted_preds)
                error_train = self.compute_error(train_full[idx_y].T.squeeze(), weighted_preds)

                print("{},{},{}".format(eps, disparity_train, error_train))
                print("{},{},{}".format(eps, disparity_train, error_train), file=f_output)

                # Q = res._asdict()["best_classifier"]
                # disp = moments.EO()
                # disp.init(train_full[idx_X], train_full[idx_A].T.squeeze(), train_full[idx_y].T.squeeze())
                #
                # error = moments.MisclassError()
                # error.init(train_full[idx_X], train_full[idx_A].T.squeeze(), train_full[idx_y].T.squeeze())
                #
                # dispVal = disp.gamma(Q).max()
                # errorVal = error.gamma(Q)[0]
                # print("{},{},{}".format(eps, dispVal, errorVal))
                # print("{},{},{}".format(eps, dispVal, errorVal), file=f_output)

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

    runner.replicate_results()
    # runner.plot_charts()

if __name__ == '__main__':
    main()
