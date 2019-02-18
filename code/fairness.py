from file_reader import FileReader

from sklearn.preprocessing import scale
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import fairlearn.moments as moments
import fairlearn.classred as red
from sklearn import cross_validation as cv
import matplotlib.pyplot as plt
import operator

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier

__author__ = 'bobo'


class PredictionFairness:

    def __init__(self):
        self.decimal = 3
        self.n_folds = 5
        self.N = 19412
        self.eps = 0.0500

        self.data_x = None
        self.data_y = None
        self.data_x_protected = None
        self.data_y_badfaith = None
        self.data_y_damaging = None
        self.data_x_attr1 = []
        self.data_y_attr1 = []
        self.data_x_attr2 = []
        self.data_y_attr2 = []

        self.label_type = 'quality'
        self.plot_output = 'dataset/plot_data_fairness'

    def load_data(self):
        reader = FileReader()
        reader.read_from_file()

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

        # create two sets of data

        # Case 1: equalize FN rates:
        # Make all the positive examples with y=1 belong to the same group (say set all of them to have a = 0)
        data_x = []
        data_y = []
        cnt_anon_pos = 0
        cnt_anon_neg = 0
        cnt_reg_pos = 0
        cnt_reg_neg = 0

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

            # Make all the positive examples with y=1 belong to the same group (a = 1)
            if self.data_y[i] == 1 and self.data_x[i][0] == 1:
                data_x.append(self.data_x[i])
                data_y.append(self.data_y[i])

            # Collect data for two attributes
            if self.data_x[i][0] == 0:
                data_x.append(self.data_x[i])
                data_y.append(self.data_y[i])

                self.data_x_attr1.append(self.data_x[i])
                self.data_y_attr1.append(self.data_y[i])

            if self.data_x[i][0] == 1:
                self.data_x_attr2.append(self.data_x[i])
                self.data_y_attr2.append(self.data_y[i])

        self.data_x = np.array(data_x)
        self.data_y = pd.Series(data_y)

        # extract the column of the protected attribute (convert to string ..)
        self.data_x_protected = pd.Series(self.data_x[:, [0]].tolist()).apply(str)
        # delete the column of the protected attribute
        self.data_x = np.delete(self.data_x, 0, 1)

        # case 2: equalize FP rates
        # todo: make all the negative examples (y=0) in the same group (set all of them to have a = 0 again)

    def run_cross_validation(self):

        it = 0
        clf_cnt = 0
        f_output = open("{}_{}.csv".format(self.plot_output, self.label_type), 'w')

        for train_idx, test_idx in cv.KFold(len(self.data_x), n_folds=self.n_folds):
            it += 1
            print("Working on Iteration {} ..".format(it))

            X_train, X_test = self.data_x[train_idx], self.data_x[test_idx]
            Y_train, Y_test = self.data_y[train_idx], self.data_y[test_idx]
            X_train_protected, X_test_protected = self.data_x_protected[train_idx], self.data_x_protected[test_idx]

            res_tuple = red.expgrad(dataX=pd.DataFrame(X_train), dataA=X_train_protected, dataY=Y_train,
                                    learner=LogisticRegression(), cons=moments.EO(), eps=self.eps)
            res = res_tuple._asdict()

            for clf in res['classifiers']:
                clf_cnt += 1

                # fn for class 1
                rate_fn_attr1 = 0
                data_x_attr1 = np.array(self.data_x_attr1)
                data_y_attr1 = np.array(self.data_y_attr1)
                for train_idx, test_idx in cv.KFold(len(self.data_x_attr1), n_folds=self.n_folds):
                    X_train, X_test = data_x_attr1[train_idx], data_x_attr1[test_idx]
                    Y_train, Y_test = data_y_attr1[train_idx], data_y_attr1[test_idx]

                    clf.fit(X_train, Y_train)
                    Y_pred = clf.predict(X_test)
                    tn, fp, fn, tp = confusion_matrix(y_true=Y_test, y_pred=Y_pred).ravel()
                    # print("cl1 FP rate: {}".format(fn / (fn + tp)))
                    rate_fn_attr1 += fn / (fn + tp)

                # fn for class 2
                rate_fn_attr2 = 0
                data_x_attr2 = np.array(self.data_x_attr2)
                data_y_attr2 = np.array(self.data_y_attr2)
                for train_idx, test_idx in cv.KFold(len(self.data_x_attr2), n_folds=self.n_folds):
                    X_train, X_test = data_x_attr2[train_idx], data_x_attr2[test_idx]
                    Y_train, Y_test = data_y_attr2[train_idx], data_y_attr2[test_idx]

                    clf.fit(X_train, Y_train)
                    Y_pred = clf.predict(X_test)
                    tn, fp, fn, tp = confusion_matrix(y_true=Y_test, y_pred=Y_pred).ravel()
                    # print("cl2 FP rate: {}".format(fn / (fn + tp)))
                    rate_fn_attr2 += fn / (fn + tp)

                # accuracy of two classes
                data_x = np.array(self.data_x_attr1 + self.data_x_attr2)
                data_y = np.array(self.data_y_attr1 + self.data_y_attr2)
                accuracy = 0
                for train_idx, test_idx in cv.KFold(len(data_x), n_folds=self.n_folds):
                    X_train, X_test = data_x[train_idx], data_x[test_idx]
                    Y_train, Y_test = data_y[train_idx], data_y[test_idx]

                    clf.fit(X_train, Y_train)
                    Y_pred = clf.predict(X_test)
                    tn, fp, fn, tp = confusion_matrix(y_true=Y_test, y_pred=Y_pred).ravel()
                    # print("accuracy rate: {}".format((tp + tn) / (fn + tn + fp + tp)))
                    accuracy += (tp + tn) / (fn + tn + fp + tp)

                print("Model {}: unfairness disparity {:.5f}, accuracy {:.5f}".format(clf_cnt,
                                                                                      abs(rate_fn_attr2 - rate_fn_attr1)/self.n_folds,
                                                                                      accuracy/self.n_folds))
                print("{},{},{}".format(clf_cnt,
                                        abs(rate_fn_attr2 - rate_fn_attr1) / self.n_folds,
                                        accuracy / self.n_folds), file=f_output)

    def plot_charts(self):
        x = []
        y = []
        d = {}
        for line in open("{}_{}.csv".format(self.plot_output, self.label_type), 'r'):
            model, unfairness, accuracy = line.strip().split(',')
            unfairness = float(unfairness)
            accuracy = float(accuracy)
            if unfairness in d:
                d[unfairness].append(accuracy)
            else:
                d[unfairness] = [accuracy]

        for key, val in d.items():
            d[key] = sum(d[key]) / len(d[key])

        for unfairness, accuracy in sorted(d.items(), key=operator.itemgetter(1)):
            y.append(unfairness)
            x.append(accuracy)

        if self.label_type == 'quality':
            # equalizing FN
            plt.ylabel('Unfairness (Disparity of False Negative Rates/Quality Control)')
            plt.xlabel('Prediction accuracy')
            plt.title('Value Trade-off between Unfairness and Prediction Accuracy\n(Editing Quality)')
        elif self.label_type == 'intention':
            # equalizing FN
            plt.ylabel('Unfairness (Disparity of False Negative Rates/Motivation Protection)')
            plt.xlabel('Prediction accuracy')
            plt.title('Value Trade-off between Unfairness and Prediction Accuracy\n(Editing Intention)')
        else:
            # TODO: equalizing FP rates
            print("Invalid prediction label ..")
            return

        plt.plot(x, y, marker='o')
        plt.show()


def main():
    runner = PredictionFairness()
    runner.load_data()
    runner.data_reformulation()
    runner.run_cross_validation()
    runner.plot_charts()

if __name__ == '__main__':
    main()
