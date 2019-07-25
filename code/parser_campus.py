import math
import numpy as np
import pandas as pd
import sys
from sklearn.model_selection import train_test_split

from datetime import datetime

class Compas:

    def __init__(self):
        self.DATA_DIR='./data/'
        self.relevant = [
            'sex',
            'race',
            'age',
            'age_cat',
            'juv_fel_count',
            'decile_score',
            'juv_misd_count',
            'juv_other_count',
            'priors_count',
            'days_b_screening_arrest',
            'c_days_from_compas',
            'v_decile_score',
            'v_score_text',
            'decile_score.1',
            'score_text',
            'c_charge_degree',
            'days_between_jail_in_out',
            'label']

        # Should we use the location-based version?
        self.files = dict(scores = 'campus-scores.csv')
        self.races = 'White Black'.split()

    def days_between(self, row):
        d2 = row['c_jail_out']
        d1 = row['c_jail_in']
        try:
            d1 = datetime.strptime(d1, "%Y-%m-%d %H:%M:%S")
            d2 = datetime.strptime(d2, "%Y-%m-%d %H:%M:%S")
            return abs((d2 - d1).days)
        except:
            sys.exit(1)

    def cleanup_frame(self, frame):
        """Make the columns have better names, and ordered in a better order"""
        frame.race = frame.race.replace({'Caucasian': 'White', \
        'African-American': 'Black'})
        frame = frame[(frame.c_charge_degree != 'O') &  # Ordinary traffic offences
                        (~frame.score_text.isnull()) &
                        (frame.is_recid != -1) &  # Missing data
                        (frame.days_b_screening_arrest.abs() <= 30)]
                        # Possibly wrong offense
        frame = frame[frame.race.isin(self.races)]
        frame['days_between_jail_in_out'] = frame.apply(lambda row: self.days_between(row),axis=1)
        frame['label'] = frame['is_recid']
        return frame[self.relevant]

    def parse_data(self):
        data_dir = 'dataset/data_campus/compas-scores.csv'
        return self.cleanup_frame(pd.read_csv(data_dir+self.files['scores']))

    def load_data(self):
        data_dir = 'dataset/data_campus/compas-scores.csv'
        # return self.cleanup_frame(pd.read_csv(data_dir+self.files['scores']))
        return self.cleanup_frame(pd.read_csv(data_dir))

    def create_train_test(self):
        random_state = np.random.RandomState(12)
        data = self.load_data()
        train_data, test_data = train_test_split(data, test_size=0.5, random_state=random_state)
        train_data = train_data.reset_index()
        test_data = test_data.reset_index()
        return train_data, test_data, data

    # create a balanced data set by race
    def adjust_data(self, data, a_type):
        if a_type == 'race':
            # select data that is only white or only black
            d_white = data[data['race'] == 1]
            d_black = data[data['race'] == 0]

            # check the number of data points in each set
            print(d_white.shape, d_black.shape)
            # (1634, 19)(2325, 19)

            # select a number and randomly select that number of data points from each set
            # import random
            # random.shuffle(d_white)
            # random.shuffle(d_black)

            n = 1500
            d_white, d_black = d_white[:n], d_black[:n]
            data = d_black.append(d_white)
            print(d_white.shape, d_black.shape, data.shape)
        else:
            d_male = data[data['sex'] == 1]
            d_female = data[data['sex'] == 0]

            # check the number of data points in each set
            print(d_male.shape, d_female.shape)
            # (1634, 19)(2325, 19)

            # select a number and randomly select that number of data points from each set
            # import random
            # random.shuffle(d_white)
            # random.shuffle(d_black)

            n = 800
            d_male, d_female = d_male[:n], d_female[:n]
            data = d_male.append(d_female)
            print(d_male.shape, d_female.shape, data.shape)
        return data

    def create_data(self, a_type):
        train, test, data = self.create_train_test()

        for col in train.columns:
            if train[col].dtype.name == 'object' or train[col].dtype.name == 'category':
                train[col] = train[col].astype('category').cat.codes
                test[col] = test[col].astype('category').cat.codes
                data[col] = data[col].astype('category').cat.codes

        features = [
            'age',
            'age_cat',
            'juv_fel_count',
            'decile_score',
            'juv_misd_count',
            'juv_other_count',
            'priors_count',
            'days_b_screening_arrest',
            'c_days_from_compas',
            'v_decile_score',
            'v_score_text',
            'decile_score.1',
            'score_text',
            'c_charge_degree',
            'days_between_jail_in_out']

        protected_attribute = [a_type]
        if a_type == 'race':
            features.insert(0, 'sex')
        elif a_type == 'sex':
            features.insert(0, 'race')
        else:
            features.insert(0, 'race')
            features.insert(0, 'sex')
        label = ['label']

        train = self.adjust_data(train, a_type)
        return train, test, data, features, protected_attribute, label, "compas"