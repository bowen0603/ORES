#!/usr/local/bin/python3

import numpy as np
import pandas as pd
import sys
from sklearn.model_selection import train_test_split


class Dutch:
    def __init__(self):
        self.positive_label = 1
        self.negative_label = 0

    def read_data(self, filename):
        data = pd.read_csv(filename, \
                           sep=r'\s*,\s*', engine='python', na_values='?')
        data['gender'] = data['sex'].map({1: "Male", 2: "Female"})
        data = data.drop(['sex'], axis=1)
        data['household_position'] = data['household_position'].map(
            {1110: "child", 1121: "married-without-children", 1122: "married-with-children",
             1131: "living-together-without-children", 1132: "living-together-with-children", 1140: "single-parent",
             1210: "single", 1220: "other-private-household", 9998: "unknown"})
        data['household_size'] = data['household_size'].map({111: 1, 112: 2, 113: 3, 114: 4, 125: 5, 126: 6})
        data['prev_residence_place'] = data['prev_residence_place'].map({1: "same", 2: "moved", 9: "unknown"})
        data['citizenship'] = data['citizenship'].map({1: "The-Netherlands", 2: "other-Europe", 3: "other"})
        data['country_birth'] = data['country_birth'].map({1: "The-Netherlands", 2: "other-Europe", 3: "other"})
        data['edu_level'] = data['edu_level'].map(
            {0: "pre-primary", 1: "primary", 2: "lower-secondary", 3: "upper-secondary", 4: "post-secondary",
             5: "tertiary", 6: "no-education"})
        data['economic_status'] = data['economic_status'].map(
            {111: "employee-other", 112: "teacher", 120: "self-employed", 210: "unemployed", 221: "student",
             222: "retired", 223: "homemaker", 224: "other"})
        data['cur_eco_activity'] = data['cur_eco_activity'].map(
            {111: "agriculture-hunting-forestry-fishing", 122: "mining-manufacturing-electricity",
             131: "wholesale-retail-repair", 132: "hotels-restaurants", 133: "transport", 134: "financial",
             135: "real-estate", 136: "public-administration", 137: "education", 138: "health-social",
             139: "other-community", 200: "not-working"})

        # 2_1: legislators, senior officials and managers; professionals
        # 5_4_9: service, shop, market sales workers; clerks; elementary occupations
        data['label'] = data['label'].map({"2_1": 1, "5_4_9": 0})

        # print(data.groupby('cur_eco_activity').label.count())
        #print(data.groupby('cur_eco_activity').label.mean())
        return data

    def load_data(self):
        return self.read_data('dataset/data_census/dutch_census_2001.csv')

    def create_train_test(self):
        random_state = np.random.RandomState(12)

        data = self.read_data('dataset/data_census/dutch_census_2001.csv')
        train_data, test_data = train_test_split(data, test_size=0.25, random_state=random_state)
        train_data = train_data.reset_index()
        test_data = test_data.reset_index()

        return train_data, test_data

    def create_data(self):
        train, test = self.create_train_test()

        for col in train.columns:
            if train[col].dtype.name == 'object' or train[col].dtype.name == 'category':
                train[col] = train[col].astype('category').cat.codes
                test[col] = test[col].astype('category').cat.codes

        features = ['age',
                    'household_position',
                    'prev_residence_place',
                    'citizenship',
                    'country_birth',
                    'edu_level',
                    'economic_status',
                    'cur_eco_activity',
                    'household_size',
                    'marital_status']

        protected_attribute = ['gender']
        label = ['label']

        return train, test, features, protected_attribute, label
