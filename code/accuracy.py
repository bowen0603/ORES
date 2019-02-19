from file_reader import FileReader

from sklearn.preprocessing import scale
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
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

        self.label_type = 'quality'

        self.plot_output = "dataset/plot_data_accuracy"

    def load_data(self):
        reader = FileReader()
        reader.read_from_file()

        # TODO: no rescaling on boolean variables..
        self.data_x = self.data_rescale(reader.data_x)
        self.data_y_damaging = reader.data_y_damaging
        self.data_y_badfaith = reader.data_y_badfaith

    def data_rescale(self, data):
        return scale(data)

    def run_cross_validation(self):

        it = 0
        # Threshold splits between [0, 1]: (1) 0, 1, 21: 0.05; (2) 0, 1, 41: 0.025
        thresholds = np.linspace(0, 1, 41, endpoint=True)
        dict_rates_fp = {}
        dict_rates_fn = {}
        dict_precision = {}
        dict_accuracy = {}
        dict_rates_tp = {}  # recall
        dict_rates_tn = {}

        for threshold in thresholds:
            threshold = str(round(threshold, self.decimal))
            dict_rates_fp[threshold] = 0
            dict_rates_fn[threshold] = 0
            dict_precision[threshold] = 0
            dict_accuracy[threshold] = 0
            dict_rates_tp[threshold] = 0
            dict_rates_tn[threshold] = 0

        for train_idx, test_idx in cv.KFold(self.N, n_folds=self.n_folds):
            it += 1
            print("Working on Iteration {} ..".format(it))

            if self.label_type == 'quality':
                data_y = self.data_y_damaging
            elif self.label_type == 'intention':
                data_y = self.data_y_badfaith
            else:
                print("Invalid prediction label ..")
                return

            # todo: include editor features here?
            X_train, X_test = self.data_x[train_idx], self.data_x[test_idx]
            Y_train, Y_test = data_y[train_idx], data_y[test_idx]

            # clf = LogisticRegression()  # default P>0.5
            # clf = AdaBoostClassifier()
            # clf = RandomForestClassifier()
            clf = MLPClassifier()

            clf.fit(X_train, Y_train)
            Y_pred_score = clf.predict_proba(X_test)

            for threshold in thresholds:
                list_Y_pred = []
                for score in Y_pred_score:
                    list_Y_pred.append(1 if score[1] >= threshold else 0)

                tn, fp, fn, tp = confusion_matrix(y_true=Y_test, y_pred=list_Y_pred).ravel()
                rate_fp = fp / (fp + tn)
                rate_tp = tp / (tp + fn)  # recall
                rate_fn = fn / (fn + tp)
                rate_tn = tn / (tn + fp)
                precision = 0 if tp + fp == 0 else tp / (tp + fp)
                accuracy = (tp + tn) / (tp + tn + fp + fn)

                threshold = str(round(threshold, self.decimal))
                dict_rates_fp[threshold] += rate_fp
                dict_rates_fn[threshold] += rate_fn

                dict_precision[threshold] += precision
                dict_rates_tp[threshold] += rate_tp
                dict_accuracy[threshold] += accuracy
                dict_rates_tn[threshold] += rate_tn

        f_output = open("{}_{}.csv".format(self.plot_output, self.label_type), 'w')
        for threshold in thresholds:
            threshold = str(round(threshold, self.decimal))
            dict_rates_fp[threshold] /= self.n_folds
            dict_rates_fn[threshold] /= self.n_folds
            # print("{}, \t{:.5f}, \t{:.5f}".format(threshold,
            #                                       dict_rates_fp[threshold],
            #                                       dict_rates_fn[threshold]))

            print("{},{:.5f},{:.5f}".format(threshold,
                                            dict_rates_fp[threshold],
                                            dict_rates_fn[threshold]), file=f_output)

            dict_precision[threshold] /= self.n_folds
            dict_rates_tp[threshold] /= self.n_folds  # recall
            dict_accuracy[threshold] /= self.n_folds
            dict_rates_tn[threshold] /= self.n_folds

            print("{} \t{:.5f} \t{:.5f} \t{:.5f}".format(threshold,
                                                         dict_precision[threshold],
                                                         dict_rates_tp[threshold],
                                                         dict_accuracy[threshold]))

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


def main():
    runner = AccuracyTradeOffs()
    runner.load_data()
    runner.run_cross_validation()
    runner.plot_charts()


if __name__ == '__main__':
    main()
