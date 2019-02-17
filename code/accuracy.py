from file_reader import FileReader

from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn import cross_validation as cv
import numpy as np

__author__ = 'bobo'

class AccuracyTradeOffs:

    def __init__(self):
        self.decimal = 3
        self.n_folds = 5
        self.N = 19412

        self.data_x = None
        self.data_y_badfaith = None
        self.data_y_damaging = None

    def load_data(self):
        reader = FileReader()
        reader.read_from_file()

        self.data_x = self.data_rescale(reader.data_x)
        self.data_y_damaging = reader.data_y_damaging
        self.data_y_badfaith = reader.data_y_badfaith

    def data_rescale(self, data):
        from sklearn.preprocessing import scale
        # scale the data set to the center
        return scale(data)

    # todo: goal is to generate the full set of data for plots
    # threshold (bad faith), fp, fn
    # threshold (damaging), fp, fn
    # possible to use the model trained by intent labels to predict damaging labels??
    def run_cross_validation(self):

        it = 0
        # Threshold splits between [0, 1]: (1) 0, 1, 21: 0.05; (2) 0, 1, 41: 0.025
        thresholds = np.linspace(0, 1, 41, endpoint=True)
        dict_rates_fp = {}
        dict_rates_fn = {}
        for threshold in thresholds:
            threshold = str(round(threshold, self.decimal))
            dict_rates_fp[threshold] = 0
            dict_rates_fn[threshold] = 0

        for train_idx, test_idx in cv.KFold(self.N, n_folds=self.n_folds):
            it += 1
            print("Working on Iteration {} ..".format(it))

            # todo: include editor features here?
            X_train, X_test = self.data_x[train_idx], self.data_x[test_idx]
            Y_train, Y_test = self.data_y_damaging[train_idx], self.data_y_damaging[test_idx]

            clf = LogisticRegression()  # default P>0.5
            # clf = AdaBoostClassifier()

            clf.fit(X_train, Y_train)
            Y_pred = clf.predict(X_test)
            Y_pred_score = clf.predict_proba(X_test)

            for threshold in thresholds:
                list_Y_pred = []
                for score in Y_pred_score:
                    list_Y_pred.append(1 if score[1] >= threshold else 0)

                tn, fp, fn, tp = confusion_matrix(y_true=Y_test, y_pred=list_Y_pred).ravel()
                rate_fp = fp / (fp + tn)
                rate_fn = fn / (fn + tp)

                threshold = str(round(threshold, self.decimal))
                dict_rates_fp[threshold] += rate_fp
                dict_rates_fn[threshold] += rate_fn

        print("Threshold, FP rate, FN rates")
        for threshold in thresholds:
            threshold = str(round(threshold, self.decimal))
            dict_rates_fp[threshold] /= self.n_folds
            dict_rates_fn[threshold] /= self.n_folds
            print("{}, \t{:.5f}, \t{:.5f}".format(threshold,
                                                  dict_rates_fp[threshold],
                                                  dict_rates_fn[threshold]))

        # TODO: train label bad faith edits
        # TODO: generate plots ...


def main():
    runner = AccuracyTradeOffs()
    runner.load_data()
    runner.run_cross_validation()


if __name__ == '__main__':
    main()