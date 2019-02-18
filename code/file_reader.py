__author__ = 'bobo'

import json
import pickle
import base64
import numpy as np


class FileReader:
    def __init__(self):
        self.filename = 'dataset/enwiki.labeled_revisions.w_cache.20k_2015.json'

        self.data_x = []
        self.N = 19412
        # todo: add protected features (registration, gender)
        # either the first two or the last two columns
        self.data_y_badfaith = []
        self.data_y_damaging = []
        self.data_rev_id = []
        self.data_protected_anon = []

    def read_from_file(self):
        cnt_line = 0
        for line in open(self.filename, 'r'):
            cnt_line += 1
            data_obj = json.loads(line)
            dict_features = pickle.loads(base64.b85decode(bytes(data_obj['cache'], 'ascii')))

            vals = []
            is_anon = 0
            for key, val in dict_features.items():
                if key == 'feature.revision.user.is_anon':
                    is_anon = 1 if val is True else 0
                    continue

                if isinstance(val, bool):
                    val = 1 if val is True else 0
                vals.append(val)

            # insert protected feature is_anon at 0
            vals.insert(0, is_anon)
            self.data_x.append(vals)

            self.data_y_damaging.append(1 if data_obj['damaging'] is True else 0)
            self.data_y_badfaith.append(0 if data_obj['goodfaith'] is True else 1)
            self.data_rev_id.append(data_obj['rev_id'])

            if cnt_line % 1000 == 0:
                print("{}k out of 19k lines processed..".format(cnt_line/1000, self.N))

        self.data_x = np.array(self.data_x)
        self.data_y_damaging = np.array(self.data_y_damaging)
        self.data_y_badfaith = np.array(self.data_y_badfaith)

    def group_identifier(self):
        # revision id -> editor id -> editor page/info
        # todo: registered or not
        # todo: female or male
        # todo: append attribute to x data
        pass

    def data_stats(self):

        cnt_damaging = 0
        cnt_badfaith = 0
        cnt_both = 0

        for i in range(len(self.data_y_badfaith)):
            if self.data_y_damaging[i] == 1 and self.data_y_badfaith[i] == 1:
                cnt_both += 1
            if self.data_y_damaging[i] == 1:
                cnt_damaging += 1
            if self.data_y_badfaith[i] == 1:
                cnt_badfaith += 1

        # 0.025551205439934062, 0.03868740984957758, 0.026272408819287038
        print("{}, {}, {}".format(cnt_both / len(self.data_y_badfaith),
                                  cnt_damaging / len(self.data_y_damaging),
                                  cnt_badfaith / len(self.data_y_badfaith)))
        # 0.04 damaging edits, 0.03 bad faith edits
        print("{} damaging edits, {} bad faith edits".format(sum(self.data_y_damaging)/len(self.data_y_damaging),
                                                             sum(self.data_y_badfaith)/len(self.data_y_badfaith)))

        # todo: % of registered or not, editors
        # todo: % of female, male, or none editors


def main():
    reader = FileReader()
    reader.read_from_file()
    reader.data_stats()

if __name__ == '__main__':
    main()