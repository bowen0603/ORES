from file_reader import FileReader

__author__ = 'bobo'

class AccuracyTradeOffs:

    def __init__(self):
        self.n_folds = 10
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
        from sklearn import cross_validation as cv

        it = 0
        for train_idx, test_idx in cv.KFold(self.N, n_folds=self.n_folds):
            it += 1

            # Split the dataset into training and test
            # todo: include editor features here?
            X_train, X_test = self.data_x[train_idx], self.data_x[test_idx]
            Y_train, Y_test = self.data_y_damaging[train_idx], self.data_y_damaging[test_idx]

            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression()

            clf.fit(X_train, Y_train)
            # Y_pred = clf.predict(X_test)
            Y_pred_score = clf.predict_proba(X_test)

            # Thresholds on bad faith edits only trained by the model with intent labels ..
            # Modeling tuning does not affect predictions of the other label ..
            import numpy as np
            # for threshold in np.arange(0.5, 1.0, 0.05):
            for threshold in np.linspace(0.5, 1, 11, endpoint=True):

                list_Y_pred = []
                for score in Y_pred_score:
                    list_Y_pred.append(1 if score[1] >= threshold else 0)

                from sklearn.metrics import confusion_matrix
                tn, fp, fn, tp = confusion_matrix(y_true=Y_test, y_pred=list_Y_pred).ravel()
                rate_fp = fp / (fp + tn)
                rate_fn = fn / (fn + tp)
                print("{}, fp: {}, fn: {}".format(threshold, rate_fp, rate_fn))

                # todo: store and compute the average values for 10 folds
                # TODO: train label bad faith edits


def main():
    runner = AccuracyTradeOffs()
    runner.load_data()
    runner.run_cross_validation()


if __name__ == '__main__':
    main()