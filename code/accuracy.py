from parser_wiki import ParserWiki
from parser_adult import Adult
from parser_dutch import Dutch
from parser_lawschool import Lawschool
from parser_campus import Campus

from sklearn.preprocessing import scale
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import cross_validation as cv

import numpy as np
import matplotlib.pyplot as plt

__author__ = 'bobo'


class AccuracyTradeOffs:

    def __init__(self):
        self.decimal = 2
        self.n_folds = 5
        self.N = 19412
        self.threshold_density = 21  # 21, 41, 81, 101

        self.df = None
        self.data_x = None
        self.data_y_badfaith = None
        self.data_y_damaging = None
        self.idx_X = None
        self.idx_A = None
        self.idx_y = None
        self.train = None
        self.test = None

        self.label_type = 'intention'

        self.plot_output = "dataset/plot_data_accuracy"

    def load_data(self):
        reader = ParserWiki()
        # reader.parse_data()
        reader.read_from_file_np()
        self.train, self.test, self.idx_X, self.idx_A, self.idx_y = reader.create_data()

        # TODO: no rescaling on boolean variables..
        # self.data_x = self.data_rescale(reader.data_x)
        self.data_x = reader.data_x
        self.data_y_damaging = reader.data_y_damaging
        self.data_y_badfaith = reader.data_y_badfaith

    def data_rescale(self, data):
        return scale(data)

    def run_cross_validation(self):

        it = 0
        thresholds = np.linspace(0, 1, self.threshold_density, endpoint=True)
        dict_rates_fp_damaging = {}
        dict_rates_fn_damaging = {}
        dict_rates_fp_badfaith = {}
        dict_rates_fn_badfaith = {}

        dict_precision_damaging = {}
        dict_accuracy_damaging = {}
        dict_sensitivity_damaging = {}  # recall
        dict_specificity_damaging = {}

        dict_precision_badfaith = {}
        dict_accuracy_badfaith = {}
        dict_sensitivity_badfaith = {}  # recall
        dict_specificity_badfaith = {}

        list_roc_auc_damaging = []
        list_pr_auc_damaging = []

        list_roc_auc_badfaith = []
        list_pr_auc_badfaith = []

        for threshold in thresholds:
            threshold = str(round(threshold, self.decimal))
            dict_rates_fp_damaging[threshold] = 0
            dict_rates_fn_damaging[threshold] = 0
            dict_rates_fp_badfaith[threshold] = 0
            dict_rates_fn_badfaith[threshold] = 0

            dict_precision_damaging[threshold] = 0
            dict_accuracy_damaging[threshold] = 0
            dict_sensitivity_damaging[threshold] = 0
            dict_specificity_damaging[threshold] = 0

            dict_precision_badfaith[threshold] = 0
            dict_accuracy_badfaith[threshold] = 0
            dict_sensitivity_badfaith[threshold] = 0
            dict_specificity_badfaith[threshold] = 0

        for train_idx, test_idx in cv.KFold(self.N, n_folds=self.n_folds):
            it += 1
            print("Working on Iteration {} ..".format(it))

            X_train, X_test = self.data_x[train_idx], self.data_x[test_idx]
            y_train_damaging, y_test_damaging = self.data_y_damaging[train_idx], self.data_y_damaging[test_idx]
            y_train_badfaith, y_test_badfaith = self.data_y_badfaith[train_idx], self.data_y_badfaith[test_idx]

            # clf = LogisticRegression(max_iter=500, penalty='l1', C=1)  # default P>0.5
            # clf = AdaBoostClassifier()
            # clf = AdaBoostClassifier(learning_rate=0.01, n_estimators=100,
            #                          base_estimator=DecisionTreeClassifier(max_depth=4, max_features="sqrt"))
            clf = GradientBoostingClassifier(learning_rate=0.01, max_depth=7, max_features="sqrt", n_estimators=700)
            # clf = RandomForestClassifier(n_estimators=500, max_features="sqrt", max_depth=7)
            # clf = MLPClassifier(learning_rate='adaptive', max_iter=500, hidden_layer_sizes=100)

            # one particular label for training
            clf.fit(X_train, y_train_damaging)
            # clf.fit(X_train, y_train_badfaith)
            y_pred_prob = clf.predict_proba(X_test)

            y_pred_score = []
            for neg, pos in y_pred_prob:
                y_pred_score.append(pos)

            list_roc_auc_damaging.append(roc_auc_score(y_true=y_test_damaging, y_score=y_pred_score))
            list_pr_auc_damaging.append(average_precision_score(y_true=y_test_damaging, y_score=y_pred_score))

            list_roc_auc_badfaith.append(roc_auc_score(y_true=y_test_badfaith, y_score=y_pred_score))
            list_pr_auc_badfaith.append(average_precision_score(y_true=y_test_badfaith, y_score=y_pred_score))

            for threshold in thresholds:
                list_Y_pred = []

                for score in y_pred_prob:
                    list_Y_pred.append(1 if score[1] >= threshold else 0)

                threshold = str(round(threshold, self.decimal))

                # Predicting damaging edits
                tn, fp, fn, tp = confusion_matrix(y_true=y_test_damaging, y_pred=list_Y_pred).ravel()
                rate_fp = fp / (fp + tn)
                rate_tp = tp / (tp + fn)  # sensitivity/recall/true positive rates
                rate_fn = fn / (fn + tp)
                rate_tn = tn / (tn + fp)  # specificity/true negative rates
                precision = 0 if tp + fp == 0 else tp / (tp + fp)
                accuracy = (tp + tn) / (tp + tn + fp + fn)

                dict_rates_fp_damaging[threshold] += rate_fp
                dict_rates_fn_damaging[threshold] += rate_fn

                dict_precision_damaging[threshold] += precision
                dict_sensitivity_damaging[threshold] += rate_tp
                dict_accuracy_damaging[threshold] += accuracy
                dict_specificity_damaging[threshold] += rate_tn

                # Predicting bad-faith edits
                tn, fp, fn, tp = confusion_matrix(y_true=y_test_badfaith, y_pred=list_Y_pred).ravel()
                rate_fp = fp / (fp + tn)
                rate_tp = tp / (tp + fn)  # sensitivity/recall/true positive rates
                rate_fn = fn / (fn + tp)
                rate_tn = tn / (tn + fp)  # specificity/true negative rates
                precision = 0 if tp + fp == 0 else tp / (tp + fp)
                accuracy = (tp + tn) / (tp + tn + fp + fn)

                dict_rates_fp_badfaith[threshold] += rate_fp
                dict_rates_fn_badfaith[threshold] += rate_fn

                dict_precision_badfaith[threshold] += precision
                dict_sensitivity_badfaith[threshold] += rate_tp
                dict_accuracy_badfaith[threshold] += accuracy
                dict_specificity_badfaith[threshold] += rate_tn

        # write fn and fp rates to the disk
        f_output = open("{}.csv".format(self.plot_output), 'w')
        for threshold in thresholds:
            threshold = str(round(threshold, self.decimal))
            dict_rates_fp_damaging[threshold] /= self.n_folds
            dict_rates_fn_damaging[threshold] /= self.n_folds
            dict_rates_fp_badfaith[threshold] /= self.n_folds
            dict_rates_fn_badfaith[threshold] /= self.n_folds

            dict_sensitivity_damaging[threshold] /= self.n_folds
            dict_specificity_damaging[threshold] /= self.n_folds

            print("{},{:.5f},{:.5f},{:.5f},{:.5f}".format(threshold,
                                                          dict_rates_fp_damaging[threshold],
                                                          dict_rates_fn_damaging[threshold],
                                                          dict_rates_fp_badfaith[threshold],
                                                          dict_rates_fn_badfaith[threshold]), file=f_output)

        # print out detailed evaluation metrics for the two labels
        print("Label (Damaging)")
        print("Threshold  Precision  Sensitivity/Recall  Specificity  Accuracy  False Positive  False Negative")
        for threshold in thresholds:
            threshold = str(round(threshold, self.decimal))
            dict_precision_damaging[threshold] /= self.n_folds
            dict_sensitivity_damaging[threshold] /= self.n_folds  # recall
            dict_accuracy_damaging[threshold] /= self.n_folds
            dict_specificity_damaging[threshold] /= self.n_folds

            print("{:.3f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}".format(float(threshold),
                                                                                  dict_precision_damaging[threshold],
                                                                                  dict_sensitivity_damaging[threshold],
                                                                                  dict_specificity_damaging[threshold],
                                                                                  dict_accuracy_damaging[threshold],
                                                                                  dict_rates_fp_damaging[threshold],
                                                                                  dict_rates_fn_damaging[threshold]))
        print("Label (Bad-faith)")
        print("Threshold  Precision  Sensitivity/Recall  Specificity  Accuracy  False Positive  False Negative")
        for threshold in thresholds:
            threshold = str(round(threshold, self.decimal))
            dict_precision_badfaith[threshold] /= self.n_folds
            dict_sensitivity_badfaith[threshold] /= self.n_folds  # recall
            dict_accuracy_badfaith[threshold] /= self.n_folds
            dict_specificity_badfaith[threshold] /= self.n_folds

            print("{:.3f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}".format(float(threshold),
                                                                                  dict_precision_badfaith[threshold],
                                                                                  dict_sensitivity_badfaith[threshold],
                                                                                  dict_specificity_badfaith[threshold],
                                                                                  dict_accuracy_badfaith[threshold],
                                                                                  dict_rates_fp_badfaith[threshold],
                                                                                  dict_rates_fn_badfaith[threshold]))

        print("Label(Damaging): ROC_AUC {}\tPR_AUC {}".format(sum(list_roc_auc_damaging) / len(list_roc_auc_damaging),
                                                              sum(list_pr_auc_damaging) / len(list_pr_auc_damaging)))
        print("Label(Bad-faith): ROC_AUC {}\tPR_AUC {}".format(sum(list_roc_auc_badfaith) / len(list_roc_auc_badfaith),
                                                               sum(list_pr_auc_badfaith) / len(list_pr_auc_badfaith)))

    def plot_charts(self):
        x = []
        y = []
        for line in open("{}_{}.csv".format(self.plot_output, self.label_type), 'r'):
            threshold, fp, fn = line.strip().split(',')
            y.append(float(fp))
            x.append(float(fn))

        # plt.xticks(rotation=90)
        plt.xticks(np.arange(min(x), max(x) + 0.1, 0.1))
        plt.yticks(np.arange(min(y), max(y) + 0.1, 0.1))

        if self.label_type == 'quality':
            plt.ylabel('FP Rate (Save Patrollers’ Efforts)')
            plt.xlabel('FN Rate (Quality Control)')
            plt.title('Value Trade-off between Save Patrollers’ Efforts and Quality Control\n(Editing Quality)')
        elif self.label_type == 'intention':
            plt.ylabel('FP Rate (Motivation Protection)')
            plt.xlabel('FN Rate (Counter-Vandalism)')
            plt.title('Value Trade-off between Motivation Protection and Counter Vandalism\n(Editing Intention)')
        else:
            print("Invalid prediction label ..")
            return

        plt.plot(x, y, marker='o')
        plt.show()

    def plot_charts2(self, filename=None):
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

        plt.xlabel('Disparity')
        plt.ylabel('Error Rate')
        # plt.plot(x, y, marker='o')
        plt.scatter(x, y, marker='o')
        plt.show()

    # Value parameterization:
    # FP_damaging: save patrollers’ efforts; FN_damaging: quality control
    # FP_badfaith: motivation protection; FN_badfaith: counter-vandalism
    def plot_all_pairs(self):

        l_fp_damaging = []
        l_fn_damaging = []
        l_fp_badfaith = []
        l_fn_badfaith = []

        for line in open("{}.csv".format(self.plot_output), 'r'):
            threshold, fp_damaging, fn_damaging, fp_badfaith, fn_badfaith = line.strip().split(',')
            l_fp_damaging.append(float(fp_damaging))
            l_fn_damaging.append(float(fn_damaging))
            l_fp_badfaith.append(float(fp_badfaith))
            l_fn_badfaith.append(float(fn_badfaith))

        f, axarr = plt.subplots(2, 3)

        axarr[0, 0].plot(l_fn_badfaith, l_fp_badfaith, marker='o')
        axarr[0, 0].set_title('Counter-Vandalism (x) V.S. \nMotivation Protection (y)')

        axarr[0, 1].plot(l_fn_damaging, l_fp_damaging, marker='o')
        axarr[0, 1].set_title('Quality Control (x) V.S. \nSave Patrollers’ Efforts (y)')

        axarr[0, 2].plot(l_fn_badfaith, l_fp_damaging, marker='o')
        axarr[0, 2].set_title('Counter-Vandalism (x) V.S. \nSave Patrollers’ Efforts (y)')

        axarr[1, 0].plot(l_fn_damaging, l_fp_badfaith, marker='o')
        axarr[1, 0].set_title('Quality Control (x) V.S. \nMotivation Protection (y)')

        axarr[1, 1].plot(l_fp_damaging, l_fp_badfaith, marker='o')
        axarr[1, 1].set_title('Save Patrollers’ Efforts (x) V.S. \nMotivation Protection (y)')

        axarr[1, 2].plot(l_fn_damaging, l_fn_badfaith, marker='o')
        axarr[1, 2].set_title('Quality Control (x) V.S. \nCounter-Vandalism (y)')

        f.subplots_adjust(hspace=0.3)
        plt.show()

    def create_individual_visuals(self):

        # threshold,idx,case
        fout = open("dataset/individual_points.csv", 'w')
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(self.data_x, self.data_y_damaging,
                                                            test_size=0.5, random_state=22)

        clf = GradientBoostingClassifier(learning_rate=0.01, max_depth=7, max_features="sqrt", n_estimators=700)
        clf.fit(X_train, y_train)
        y_pred_prob = clf.predict_proba(X_test)

        idx = 0
        thresholds = np.linspace(0, 1, self.threshold_density, endpoint=True)
        print(len(y_pred_prob))
        cnt1 = 0
        cnt2 = 0
        tp = fp = fn = tn = 0
        for y_score in y_pred_prob:
            if X_test[idx][0] == 0.0:
                cnt1 += 1

            if X_test[idx][0] == 1.0:
                cnt2 += 1
            for threshold in thresholds:
                predicted_cls = 1 if y_score[1] >= threshold else 0
                if predicted_cls == 1 and y_test[idx] == 1:
                    case = 0  # tp (correctly predicted bad faith edits)
                    tp += 1
                elif predicted_cls == 1 and y_test[idx] == 0:
                    case = 1  # fp (good edits predicted as bad faith edits)
                    fp += 1
                elif predicted_cls == 0 and y_test[idx] == 1:
                    case = 2  # fn (bad edits predicted as good faith edits)
                    fn += 1
                else:
                    case = 3  # tn (correctly predicted good faith edits)
                    tn += 1

                threshold = str(round(threshold, self.decimal))
                print("{},{},{},{}".format(idx, threshold, case, int(X_test[idx][0])), file=fout)
            idx += 1

        print(tp, fp, fn, tn)
        print(cnt1, cnt2)

    def create_individual_visuals_bar(self):

        # threshold,idx,case
        fout = open("dataset/individual_points_bar.csv", 'w')
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(self.data_x, self.data_y_damaging,
                                                            test_size=0.5, random_state=22)

        clf = GradientBoostingClassifier(learning_rate=0.01, max_depth=7, max_features="sqrt", n_estimators=700)
        clf.fit(X_train, y_train)
        y_pred_prob = clf.predict_proba(X_test)

        idx = 0
        thresholds = np.linspace(0, 1, self.threshold_density, endpoint=True)
        print(len(y_pred_prob))

        for threshold in thresholds:

            idx = 0
            tp, fp, fn, tn = 0, 0, 0, 0
            tp_a0, fp_a0, fn_a0, tn_a0 = 0, 0, 0, 0
            for y_score in y_pred_prob:

                predicted_cls = 1 if y_score[1] >= threshold else 0
                if predicted_cls == 1 and y_test[idx] == 1:
                    case = 0  # tp
                    tp += 1
                    if X_test[idx][0] == 0.0:
                        tp_a0 += 1
                elif predicted_cls == 1 and y_test[idx] == 0:
                    case = 1  # fp
                    fp += 1
                    if X_test[idx][0] == 0.0:
                        fp_a0 += 1
                elif predicted_cls == 0 and y_test[idx] == 1:
                    case = 2  # fn
                    fn += 1
                    if X_test[idx][0] == 0.0:
                        fn_a0 += 1
                else:
                    case = 3  # tn
                    tn += 1
                    if X_test[idx][0] == 0.0:
                        tn_a0 += 1

                idx += 1

            rate_fp = 0 if fp + tn == 0 else fp / (fp + tn)
            rate_tp = 0 if tp + fn == 0 else tp / (tp + fn)  # sensitivity/recall/true positive rates
            rate_fn = 0 if fn + tp == 0 else fn / (fn + tp)
            rate_tn = 0 if fp + tn == 0 else tn / (tn + fp)  # specificity/true negative rates

            rate_fp_a0 = 0 if fp == 0 else fp_a0 / fp
            rate_tp_a0 = 0 if tp == 0 else tp_a0 / tp
            rate_fn_a0 = 0 if fn == 0 else fn_a0 / fn
            rate_tn_a0 = 0 if tn == 0 else tn_a0 / tn

            print("{},{},{},{},{},{},{},{},{}".format(threshold,
                                                        rate_tp, rate_tp_a0,
                                                        rate_fp, rate_fp_a0,
                                                        rate_fn, rate_fn_a0,
                                                        rate_tn, rate_tn_a0), file=fout)

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

    def run_train_test_split_baseline2(self):
        # Adult(), Campus(), Dutch(), Lawschool(), ParserWiki()
        for obj in [Adult()]:

            train, test, idx_X, idx_A, idx_y, data_name = obj.create_data()
            print(data_name, train.shape, test.shape, len(idx_X), len(idx_A), len(idx_y))

            # train the model using the train data with full features
            clf = GradientBoostingClassifier(learning_rate=0.01, max_depth=7, max_features="sqrt", n_estimators=700)
            # clf = RandomForestClassifier()
            clf = LogisticRegression()
            # todo: check the difference with and without idx_A
            clf.fit(train[idx_X], train[idx_y])
            y_pred_prob = clf.predict_proba(test[idx_X])

            thresholds = np.linspace(0, 1, self.threshold_density, endpoint=True)
            print(len(y_pred_prob))
            filename='dataset/plot.csv'
            f_output = open(filename, 'w')
            for threshold in thresholds:
                y_prob = y_pred_prob[:, 1].copy()

                y_prob[y_prob >= threshold] = 1
                y_prob[y_prob < threshold] = 0

                error_train = self.compute_error(train[idx_y[0]], y_prob)
                disparity_train = self.compute_FP(train[idx_A[0]], train[idx_y[0]], y_prob)

                print("{},{},{}".format(threshold, disparity_train, error_train))
                print("{},{},{}".format(threshold, disparity_train, error_train), file=f_output)

            f_output.close()
            self.plot_charts2(filename)

def main():
    runner = AccuracyTradeOffs()
    # runner.load_data()
    # runner.run_cross_validation()
    # runner.plot_charts()
    # runner.plot_all_pairs()
    # runner.create_individual_visuals()
    # runner.create_individual_visuals_bar()
    runner.run_train_test_split_baseline2()

if __name__ == '__main__':
    main()
