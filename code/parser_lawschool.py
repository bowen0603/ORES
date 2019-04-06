"""Parsing information for law school dataset"""

from __future__ import print_function
import pandas
import sys
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype

DATA_FILE = './data/BP_BASE.DAT'

class Parser(object):
    def __init__(self, name, length, *args):
        self.name = name
        self.length = length
        self.args = args

    def parse(self, s):
        return NotImplementedError

class BinaryField(Parser):
    dtype=float
    def parse(self, s):
        if not s.strip():
            return None
        try:
            return self.args[0].index(s)
        except ValueError:
            return None

class IntField(Parser):
    dtype=float
    def parse(self, s):
        if not s.strip():
            return None
        return int(s)

class CategoryField(Parser):
    dtype='category'
    def parse(self, s):
        if self.args:
            if s in self.args[0]:
                return self.args[0][s]
        if not s.strip():
            return None
        return s

class StrField(Parser):
    dtype=str
    def parse(self, s):
        return s

class FloatField(Parser):
    dtype=float
    def parse(self, s):
        if s.strip() == '.':
            return None
        return float(s)

class DateField(Parser):
    dtype=object
    def parse(self, s):
        if not s.strip():
            return None
        Y = int(s[:2])
        if not s[2:].strip():
            M = None
        else:
            M = int(s[2:])
        if self.args and self.args[0] == True: # MMYY not YYMM.
            Y, M = M, Y
        return Y, M

class Lawschool:
    def __init__(self):
        self.race = 'White Black'.split()
        self.race_coding = {str(i+1): s for i, s in enumerate('Amerindian Asian Black Mexican Puertorican Hispanic White Other'.split())}
        self.data_format = [
            IntField('id', 5),
            CategoryField('sex', 1),
            CategoryField('race', 1, self.race_coding),
            IntField('origin_cluster', 1),
            IntField('degree_cluster', 1),
            FloatField('LSAT', 4),
            FloatField('UGPA', 3),
            StrField('', 1), #Unused column
            FloatField('ZFYA', 5),
            DateField('birth_date', 4, True), # reversed
            IntField('major', 1),
            CategoryField('graduated', 1),
            DateField('graduation_date', 4), #YYMM
            DateField('expected_graduation_date', 4), #YYMM
            FloatField('Zcum', 5), # Cumulative final LGPA standardized within school
            CategoryField('region_first', 2),
            CategoryField('region_last', 2),
            IntField('attempts', 1),
            BinaryField('first_pf', 1, 'FP'),
            DateField('first_pf_date', 4),
            BinaryField('last_pf', 1, 'FP'),
            DateField('last_pf_date', 4),
            ]

    def parse_line_to_dict(self, s):
        cur_loc = 0
        ans = {}
        for field in self.data_format:
            value = field.parse(s[cur_loc:cur_loc+field.length])
            ans[field.name] = value
            cur_loc += field.length
        del ans['']
        return ans


    def convert_date(self, row,field):
      try:
          (year,month) = row[field]
          return year + float(month)/12
      except:
          return 95

    def print_stats(self, frame):
        print("Race value counts:")
        print(frame.race.value_counts())
        print("Last_pf value counts:")
        print(frame.last_pf.value_counts())
        print("First_pf value counts:")
        print(frame.first_pf.value_counts())

        print('Average academic indices:')
        print(frame.groupby('race').mean()['sander_index'].loc[self.race])
        print(frame.groupby('race').mean()['last_pf'].loc[self.race])
        print(frame.groupby('race').mean()['first_pf'].loc[self.race])

    def parse_file(self, filename):
        with open(filename) as f:
            rows = list(map(self.parse_line_to_dict, f))

        series = {}
        for field in self.data_format:
            if field.name == '':
                continue
            series[field.name] = pandas.Series([r[field.name] for r in rows], dtype=field.dtype)
        data = pandas.DataFrame(data=series)
        # Academic index used by Sander
        data['sander_index'] = data['LSAT']/data['LSAT'].max() * 0.6 + data['UGPA']/data['UGPA'].max() * 0.4
        data = data[data.race.isin(self.race)]
        data.race = data.race.astype(CategoricalDtype(categories=self.race))
        data.race = data.race.cat.remove_unused_categories()
        data = data[(~data.last_pf.isnull())]
        data['label'] = data['last_pf'].astype(int)
        self.print_stats(data)
        data = data.drop(['last_pf_date','last_pf','first_pf_date','first_pf','expected_graduation_date','id','graduation_date','graduated'],axis=1)
        data['birth_date'] = \
            data.apply(lambda row:self.convert_date(row,'birth_date'),axis=1)
        data = data[(~data.sex.isnull())]
        for i in data.columns[data.isna().any()].tolist():
            data[i] = data[i].fillna(data[i].mean())
        return data

    def load_data(self):
        return self.parse_file(filename='dataset/data_lawschool/BP_BASE.DAT')

    def create_train_test(self):
        data = self.parse_file(filename='dataset/data_lawschool/BP_BASE.DAT')
        random_state = np.random.RandomState(12)
        train_data, test_data = train_test_split(data,test_size=0.25,random_state=random_state)
        train_data = train_data.reset_index()
        test_data = test_data.reset_index()
        return train_data, test_data

    def create_data(self):
        train, test = self.create_train_test()

        for col in train.columns:
            if train[col].dtype.name == 'object' or train[col].dtype.name == 'category':
                train[col] = train[col].astype('category').cat.codes
                test[col] = test[col].astype('category').cat.codes

        features = ['LSAT',
                    'UGPA',
                    'ZFYA',
                    'Zcum',
                    'attempts',
                    'birth_date',
                    'degree_cluster',
                    'major',
                    'origin_cluster',
                    'region_first',
                    'region_last',
                    'sex',
                    'sander_index']

        protected_attribute = ['race']
        label = ['label']

        return train, test, features, protected_attribute, label
