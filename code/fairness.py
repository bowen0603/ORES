from file_reader import FileReader

from sklearn.preprocessing import scale
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import fairlearn.moments as moments
import fairlearn.classred as red
from sklearn import cross_validation as cv

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier

__author__ = 'bobo'


class PredictionFairness:

    def __init__(self):
        self.decimal = 3
        self.n_folds = 5
        self.N = 19412

        self.data_x = None
        self.data_x_protected = None
        self.data_y_badfaith = None
        self.data_y_damaging = None
        self.data_x_cl1 = []
        self.data_y_cl1 = []
        self.data_x_cl2 = []
        self.data_y_cl2 = []

        self.label_type = 'intention'

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

        for i in range(len(self.data_x)):

            if self.data_y_damaging[i] == 1 and self.data_x[i][0] == 1:
                cnt_anon_pos += 1
            if self.data_y_damaging[i] == 0 and self.data_x[i][0] == 1:
                cnt_anon_neg += 1
            if self.data_y_damaging[i] == 1 and self.data_x[i][0] == 0:
                cnt_reg_pos += 1
            if self.data_y_damaging[i] == 0 and self.data_x[i][0] == 0:
                cnt_reg_neg += 1

            if self.data_y_damaging[i] == 1 and self.data_x[i][0] == 1:
                data_x.append(self.data_x[i])
                data_y.append(self.data_y_damaging[i])
            if self.data_x[i][0] == 0:
                data_x.append(self.data_x[i])
                data_y.append(self.data_y_damaging[i])

                self.data_x_cl1.append(self.data_x[i])
                self.data_y_cl1.append(self.data_y_damaging[i])

            if self.data_x[i][0] == 1:
                self.data_x_cl2.append(self.data_x[i])
                self.data_y_cl2.append(self.data_y_damaging[i])

        # unregistered, damaging: 481
        # unregistered, good: 3007
        # registered, damaging: 270
        # registered, good: 15654
        print(cnt_anon_pos, cnt_anon_neg, cnt_reg_pos, cnt_reg_neg)
        # Just use the one with the more data points to hack
        self.data_x = np.array(data_x)
        self.data_y_damaging = pd.Series(data_y)

        # extract the column of the protected attribute (convert to string ..)
        self.data_x_protected = pd.Series(self.data_x[:, [0]].tolist()).apply(str)
        # delete the column of the protected attribute
        self.data_x = np.delete(self.data_x, 0, 1)

        # case 2: equalize FP rates
        # todo: make all the negative examples (y=0) in the same group (set all of them to have a = 0 again)

    def create_learners(self):
        list_learners = []

        list_learners.append(LogisticRegression())
        list_learners.append(AdaBoostClassifier())
        return list_learners

    def build_classifiers(self):

        it = 0
        for train_idx, test_idx in cv.KFold(len(self.data_x), n_folds=self.n_folds):
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
            Y_train, Y_test = self.data_y_damaging[train_idx], self.data_y_damaging[test_idx]
            X_train_protected, X_test_protected = self.data_x_protected[train_idx], self.data_x_protected[test_idx]

            learners = self.create_learners()
            res_tuple = red.expgrad(dataX=pd.DataFrame(X_train), dataA=X_train_protected, dataY=Y_train,
                                    learner=LogisticRegression(), cons=moments.EO())  # todo: tune eps

            res = res_tuple._asdict()
            Q = res["best_classifier"]
            res["n_classifiers"] = len(res["classifiers"])

            clf_cnt = 0
            for clf in res['classifiers']:
                # fn of each class
                clf_cnt += 1

                rate_fn_cl1 = 0
                data_x_cl1 = np.array(self.data_x_cl1)
                data_y_cl1 = np.array(self.data_y_cl1)
                for train_idx, test_idx in cv.KFold(len(self.data_x_cl1), n_folds=self.n_folds):
                    X_train, X_test = data_x_cl1[train_idx], data_x_cl1[test_idx]
                    Y_train, Y_test = data_y_cl1[train_idx], data_y_cl1[test_idx]

                    clf.fit(X_train, Y_train)
                    Y_pred = clf.predict(X_test)
                    tn, fp, fn, tp = confusion_matrix(y_true=Y_test, y_pred=Y_pred).ravel()
                    # print("cl1 FP rate: {}".format(fn / (fn + tp)))
                    rate_fn_cl1 += fn / (fn + tp)

                rate_fn_cl2 = 0
                data_x_cl2 = np.array(self.data_x_cl2)
                data_y_cl2 = np.array(self.data_y_cl2)
                for train_idx, test_idx in cv.KFold(len(self.data_x_cl2), n_folds=self.n_folds):
                    X_train, X_test = data_x_cl2[train_idx], data_x_cl2[test_idx]
                    Y_train, Y_test = data_y_cl2[train_idx], data_y_cl2[test_idx]

                    clf.fit(X_train, Y_train)
                    Y_pred = clf.predict(X_test)
                    tn, fp, fn, tp = confusion_matrix(y_true=Y_test, y_pred=Y_pred).ravel()
                    # print("cl2 FP rate: {}".format(fn / (fn + tp)))
                    rate_fn_cl2 += fn / (fn + tp)

                # total accuracy
                data_x = np.array(self.data_x_cl1 + self.data_x_cl2)
                data_y = np.array(self.data_y_cl1 + self.data_y_cl2)
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
                                                                                      abs(rate_fn_cl2 - rate_fn_cl1)/self.n_folds,
                                                                                      accuracy/self.n_folds))


def main():
    runner = PredictionFairness()
    runner.load_data()
    runner.data_reformulation()
    runner.build_classifiers()

if __name__ == '__main__':
    main()
