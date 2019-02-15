__author__ = 'bobo'

class AccuracyTradeOffs:

    def __init__(self):
        self.n_folds = 10
        self.n = 0 # todo

        self.data_x = None
        self.data_y_badfaith = None
        self.data_y_damaging = None

    def load_data(self):
        pass

    def data_preprocessing(self, data):
        from sklearn.preprocessing import normalize, scale
        # scale the data set to the center
        data = scale(data, with_mean=True, with_std=True, copy=True)

        # normalize the data set
        # self.x_data = normalize(self.x_data, norm='l2')
        return data

    # todo: goal is to generate the full set of data for plots
    # threshold (bad faith), fp, fn
    # threshold (damaging), fp, fn
    # possible to use the model trained by intent labels to predict damaging labels??
    def run_cross_validation(self):
        from sklearn import cross_validation as cv

        it = 0
        for train_idx, test_idx in cv.KFold(self.n, n_folds=self.n_folds):
            it += 1

            # TODO: train label bad faith edits

            ## Split the dataset into training and test
            # todo: need to handle this datasets properly .. protected attributes, etc
            X_train, X_test = self.data_x[train_idx], self.data_x[test_idx]
            Y_train, Y_test = self.data_y[train_idx], self.data_y[test_idx]

            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression()

            clf.fit(X_train, Y_train)
            Y_pred = clf.predict(X_test)

            Y_pred_score = clf.predict_proba(X_test)

            # Thresholds on bad faith edits only trained by the model with intent labels ..
            # Modeling tuning does not affect predictions of the other label ..
            import numpy as np
            # for threshold in np.arange(0.5, 1.0, 0.05):
            for threshold in np.linspace(0.5, 1, 11, endpoint=True):

                from sklearn.metrics import confusion_matrix
                tn, fp, fn, tp = confusion_matrix(y_true=[0, 1, 0, 1], y_pred=[1, 1, 1, 0]).ravel()
                # todo: convert FP, FN to rates
                # todo: store and compute the average values for 10 folds


            # TODO: train label damaging edits