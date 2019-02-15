__author__ = 'bobo'

class PredictionFairness:

    def __init__(self):
        pass

    # todo: this has to be done after the protected attributes are collected

    # todo: split into training and testing

    # todo: train the data using multiple learners from learnskit

    # todo: return the data for different pareto curve / prediction errors

    def load_data(self):
        pass

    def retrieve_editor_info(self):
        # todo: based on the edit info, identify the registration and gender info of the editors ...
        pass

    def data_preprocessing(self, data):
        from sklearn.preprocessing import normalize, scale
        # scale the data set to the center
        data = scale(data, with_mean=True, with_std=True, copy=True)

        # normalize the data set
        # self.x_data = normalize(self.x_data, norm='l2')
        return data

    def data_reformulation(self):

        # create two sets of data

        # case 1: equalize FN rates:
        # todo: make all the positive examples with y=1 belong to the same group (say set all of them to have a = 0)

        # case 2: equalize FP rates
        # todo: make all the negative examples (y=0) in the same group (set all of them to have a = 0 again)

        # split into training and testing ...
        pass

    def build_classifiers(self):

        # todo: split the data into training and testing again ..
        # todo: 1. create a set of classifiers
        # todo: 2. run the fairness code and get returned classifiers (with eps value limit)

        # todo: 3. for each model, use the training data, to compute the disparity between groups and error,
        # as one data point to plot out

        # then, repeat the process with another case of data split for equalizing FP rates...
        pass

# 1) run the algorithm on the training data, with a range eps values, as you said
# 2) take the set of output models and evaluate on the training set, calculate the false-positive/negative rate disparity between the two groups
# 3) Then you could plot the (disparity, error) of all the models, and trace out the pareto curve. But you should not take the input eps as the disparity measure.
# 4) Finally, you could evaluate the models on the training pareto curve on the test set: plotting (disparity, error) as well.


# Also, there is a hack that allows you to directly enforce equalizing false-positive rates (as opposed to both FP and FN rates).
# Recall that to equalize FP across means that
# Pr[y-hat = 1| y = 0, a = 1] = Pr[y-hat = 1| y = 0, a = 0]
# where y-hat denotes the prediction and a denotes the group membership.
#
# So you want the algorithm to ignore the constraint of equalizing FN rates:
# Pr[y-hat = 0| y = 1, a = 1] = Pr[y-hat = 0 | y = 1, a = 0]
#
# To do that you can make all the positive examples with y=1 belong to the same group (say set all of them to have a = 0).
# Then enforcing the "EO" constraint will help you enforce FP constraint only.


# Similarly, to only enforce false-negative rates, I will have to make all the negative examples (y=0) in the same group (set all of them to have a = 0 again).
# In this case, we will have to sets of training data for these two cases.