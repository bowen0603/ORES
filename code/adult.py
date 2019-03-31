#!/usr/local/bin/python3

import numpy as np
import pandas as pd
import sys
from sklearn.model_selection import train_test_split


class Adult:
    def __init__(self):
        self.names = [
            'age',
            'workclass',
            'fnlwgt',
            'education',
            'education-num',
            'marital-status',
            'occupation',
            'relationship',
            'race',
            'gender',
            'capital-gain',
            'capital-loss',
            'hours-per-week',
            'native-country',
            'label'
        ]

        self.relevant = [
            'marital-status',
            'occupation',
            'relationship',
            'race',
            'gender',
            'over_25',
            'age',
            'education-num',
            'capital-gain',
            'capital-loss',
            'hours-per-week',
            'label']

        self.positive_label = 1
        self.negative_label = 0
        self.Predefined = True

    def read_data(self, filename):
        data = pd.read_csv(filename, names=self.names, sep=r'\s*,\s*', engine='python', na_values='?')
        data['label'] = \
            data['label'].map({'<=50K': self.negative_label, '>50K': self.positive_label})
        data['over_25'] = np.where(data['age'] >= 25, 'yes', 'no')
        # data['gender'] = np.where(data['gender'] == 'Male', '1', 0)
        return data

    def create_train_test(self):
        args = {}
        args['algorithm'] = 'logistic_regression'
        args['seed'] = 12
        random_state = np.random.RandomState(args['seed'])

        if self.Predefined:
            train_data = self.read_data('fairness-data/adult/data/adult.data')
            test_data = self.read_data('fairness-data/adult/data/adult.test')
        else:
            data = self.read_data('fairness-data/adult/data/adult.all')
            train_data, test_data = train_test_split(data, test_size=0.25, random_state=random_state)
            train_data = train_data.reset_index()
            test_data = test_data.reset_index()

        if args['algorithm'] == 'logistic_regression':
            return (train_data[self.relevant], test_data[self.relevant])
        else:
            return (train_data, test_data)
