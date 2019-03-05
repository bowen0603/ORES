from file_reader import FileReader

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
        self.decimal = 3
        self.n_folds = 5
        self.N = 19412

        self.data_x = None
        self.data_y_badfaith = None
        self.data_y_damaging = None

        # self.label_type = 'quality'
        self.label_type = 'intention'

        self.plot_output = "dataset/plot_data_accuracy"

    def load_data(self):
        reader = FileReader()
        reader.read_from_file()

        # TODO: no rescaling on boolean variables..
        # self.data_x = self.data_rescale(reader.data_x)
        self.data_x = reader.data_x
        self.data_y_damaging = reader.data_y_damaging
        self.data_y_badfaith = reader.data_y_badfaith

    def data_rescale(self, data):
        return scale(data)

    def run_cross_validation(self):

        it = 0
        # Threshold splits between [0, 1]: (1) 0, 1, 21: 0.05; (2) 0, 1, 41: 0.025
        thresholds = np.linspace(0, 1, 101, endpoint=True)
        dict_rates_fp_damaging = {}
        dict_rates_fn_damaging = {}
        dict_rates_fp_badfaith = {}
        dict_rates_fn_badfaith = {}
        dict_precision = {}
        dict_accuracy = {}
        dict_sensitivity = {}  # recall
        dict_specificity = {}

        list_roc_auc = []
        list_pr_auc = []

        for threshold in thresholds:
            threshold = str(round(threshold, self.decimal))
            dict_rates_fp_damaging[threshold] = 0
            dict_rates_fn_damaging[threshold] = 0
            dict_rates_fp_badfaith[threshold] = 0
            dict_rates_fn_badfaith[threshold] = 0

            dict_precision[threshold] = 0
            dict_accuracy[threshold] = 0
            dict_sensitivity[threshold] = 0
            dict_specificity[threshold] = 0

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

            clf.fit(X_train, y_train_damaging)
            # clf.fit(X_train, y_train_badfaith)
            Y_pred_prob = clf.predict_proba(X_test)

            Y_pred_score = []
            for neg, pos in Y_pred_prob:
                Y_pred_score.append(pos)

            list_roc_auc.append(roc_auc_score(y_true=y_test_damaging, y_score=Y_pred_score))
            list_pr_auc.append(average_precision_score(y_true=y_test_damaging, y_score=Y_pred_score))

            for threshold in thresholds:
                list_Y_pred = []

                for score in Y_pred_prob:
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

                dict_precision[threshold] += precision
                dict_sensitivity[threshold] += rate_tp
                dict_accuracy[threshold] += accuracy
                dict_specificity[threshold] += rate_tn

                # Predicting bad-faith edits
                tn, fp, fn, tp = confusion_matrix(y_true=y_test_badfaith, y_pred=list_Y_pred).ravel()
                rate_fp = fp / (fp + tn)
                rate_tp = tp / (tp + fn)  # sensitivity/recall/true positive rates
                rate_fn = fn / (fn + tp)
                rate_tn = tn / (tn + fp)  # specificity/true negative rates
                precision = 0 if tp + fp == 0 else tp / (tp + fp)
                accuracy = (tp + tn) / (tp + tn + fp + fn)

                # threshold = str(round(threshold, self.decimal))
                dict_rates_fp_badfaith[threshold] += rate_fp
                dict_rates_fn_badfaith[threshold] += rate_fn

        f_output = open("{}_{}.csv".format(self.plot_output, self.label_type), 'w')
        print("Threshold  Precision  Sensitivity/Recall  Specificity  Accuracy  False Positive  False Negative")
        for threshold in thresholds:
            threshold = str(round(threshold, self.decimal))
            dict_rates_fp_damaging[threshold] /= self.n_folds
            dict_rates_fn_damaging[threshold] /= self.n_folds
            dict_rates_fp_badfaith[threshold] /= self.n_folds
            dict_rates_fn_badfaith[threshold] /= self.n_folds
            # print("{}, \t{:.5f}, \t{:.5f}".format(threshold,
            #                                       dict_rates_fp[threshold],
            #                                       dict_rates_fn[threshold]))

            print("{},{:.5f},{:.5f},{:.5f},{:.5f}".format(threshold,
                                                          dict_rates_fp_damaging[threshold],
                                                          dict_rates_fn_damaging[threshold],
                                                          dict_rates_fp_badfaith[threshold],
                                                          dict_rates_fn_badfaith[threshold]), file=f_output)

            dict_precision[threshold] /= self.n_folds
            dict_sensitivity[threshold] /= self.n_folds  # recall
            dict_accuracy[threshold] /= self.n_folds
            dict_specificity[threshold] /= self.n_folds

            print("{:.3f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}".format(float(threshold),
                                                                                  dict_precision[threshold],
                                                                                  dict_sensitivity[threshold],
                                                                                  dict_specificity[threshold],
                                                                                  dict_accuracy[threshold],
                                                                                  dict_rates_fp_damaging[threshold],
                                                                                  dict_rates_fn_damaging[threshold]))

        print("ROC_AUC {}\tPR_AUC {}".format(sum(list_roc_auc) / len(list_roc_auc),
                                             sum(list_pr_auc) / len(list_pr_auc)))

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

        # FP_damaging False-positive rate of predicting damaging edits: save patrollers’ efforts
        # FN_damaging false-negative rate of predicting damaging edits: quality control
        # FP_badfaith False-positive rate of predicting bad faith edits: motivation protection
        # FN_badfaith False-negative rate of predicting bad faith edits: counter-vandalism

        # Six polar axes
        f, axarr = plt.subplots(2, 3)

        # TODO: double check the value-param mapping

        axarr[0, 0].plot(l_fp_badfaith, l_fn_badfaith, marker='o')
        axarr[0, 0].set_title('Motivation Protection (x) V.S. \nCounter-Vandalism (y)')

        axarr[0, 1].plot(l_fp_damaging, l_fn_damaging, marker='o')
        axarr[0, 1].set_title('Save Patrollers’ Efforts (x) V.S. \nQuality Control (y)')

        axarr[0, 2].plot(l_fp_damaging, l_fn_badfaith, marker='o')
        axarr[0, 2].set_title('Save Patrollers’ Efforts (x) V.S. \nCounter-Vandalism (y)')

        axarr[1, 0].plot(l_fp_badfaith, l_fn_damaging, marker='o')
        axarr[1, 0].set_title('Motivation Protection (x) V.S. \nQuality Control (y)')

        axarr[1, 1].plot(l_fp_badfaith, l_fp_damaging, marker='o')
        axarr[1, 1].set_title('Motivation Protection (x) V.S. \nSave Patrollers’ Efforts (y)')

        axarr[1, 2].plot(l_fn_badfaith, l_fn_damaging, marker='o')
        axarr[1, 2].set_title('Counter-Vandalism (x) V.S. \nQuality Control (y)')
        # Fine-tune figure; make subplots farther from each other.
        f.subplots_adjust(hspace=0.3)

        plt.show()


def main():
    runner = AccuracyTradeOffs()
    runner.load_data()
    runner.run_cross_validation()
    # runner.plot_charts()
    runner.plot_all_pairs()


if __name__ == '__main__':
    main()
