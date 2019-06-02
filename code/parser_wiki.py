__author__ = 'bobo'

import requests
import json
import pickle
import base64
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class ParserWiki:
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
        self.data = []
        self.df = None

    def read_from_file_np(self):
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

    def is_new_editor(self, revid):

        query = "https://en.wikipedia.org/w/api.php?action=query&format=json&prop=revisions&rvprop=userid|user|timestamp&revids="+str(revid)
        response = requests.get(query).json()
        try:
            for page in response['query']['pages']:
                rev = response['query']['pages'][page]['revisions'][0]
                userid = rev['userid']
                username = rev['user']
                timestamp = rev['timestamp']

                try:
                    url_usercontb = "https://en.wikipedia.org/w/api.php?action=query&format=json&list=usercontribs&"
                    const_max_requests = 100
                    query = url_usercontb + "uclimit=" + str(const_max_requests) + \
                            "&ucprop=title|timestamp|flags|ids&ucstart=" + timestamp + "&ucdir=older&ucuser=" + username
                    response = requests.get(query).json()
                    contribs = response['query']['usercontribs']

                except requests.exceptions.ConnectionError:
                    print("Max retries exceeded with url.")
                    from time import sleep
                    sleep(5)
                    return self.is_new_editor(revid)
                #  True: newcomer; False: experienced
                # print(username, len(contribs))
                return len(contribs) < 100
        except:
            return None

    def parse_data(self):
        cnt_line = 0

        data_true = []
        data_false = []
        d_new = {}
        cnt = 0
        for line in open(self.filename, 'r'):
            cnt_line += 1
            data_obj = json.loads(line)
            dict_features = pickle.loads(base64.b85decode(bytes(data_obj['cache'], 'ascii')))
            # print(dict_features.head(5))

            vals = []
            if not data_obj['rev_id'] in d_new:
                is_new = self.is_new_editor(data_obj['rev_id'])
                if is_new is None:
                    continue
                d_new[data_obj['rev_id']] = is_new
            else:
                is_new = d_new[data_obj['rev_id']]
            is_new = 1 if is_new is True else 0

            is_anon = 0
            for key, val in dict_features.items():
                if key == 'feature.revision.user.is_anon':
                    is_anon = 1 if val is True else 0
                    continue

                if isinstance(val, bool):
                    val = 1 if val is True else 0
                vals.append(val)

            # data structure: insert (label 1 (faith), label 2 (damaging), A_anon, A_new, ...)
            # insert protected feature is_anon at 0
            vals.insert(0, is_new)
            vals.insert(0, is_anon)
            vals.insert(0, 1 if data_obj['damaging'] is True else 0)
            vals.insert(0, 0 if data_obj['goodfaith'] is True else 1)

            # if data_obj['damaging']:
            #     self.data.append(vals)  # 751

            if not data_obj['damaging']:
                cnt += 1
                if cnt == 3000:
                    break
                self.data.append(vals)  # 751

            # if not data_obj['goodfaith']:
            #     self.data.append(vals)   # 510

            # if data_obj['goodfaith']:
            #     self.data.append(vals)  # 510

            # self.data.append(vals)

            if cnt_line % 1000 == 0:
                print("{}k out of 19k lines processed..".format(cnt_line/1000, self.N))

        print(cnt)
        # return None
        self.data_x = np.array(self.data_x)
        self.data_y_damaging = np.array(self.data_y_damaging)
        self.data_y_badfaith = np.array(self.data_y_badfaith)

        import random
        random.shuffle(self.data)
        # self.df = pd.DataFrame(data=self.data[:3000])
        self.df = pd.DataFrame(data=self.data)
        self.df.to_csv('dataset/enwiki_new_non_dam.csv', index=False, header=False)

        return self.df

    def create_data(self):
        self.df = pd.read_csv('dataset/enwiki_new_balanced.csv', header=None)
        self.df = self.df.sample(frac=1).reset_index(drop=True)
        train, test = train_test_split(self.df, test_size=0.5, random_state=12)

        idx_faith = [0]
        idx_damaging = [1]
        idx_A = [2, 3]
        idx_X = list(range(4, 83))

        # dd = self.df.loc[self.df[idx_damaging] == 1]
        # df_damaging = self.df.loc[self.df[idx_damaging] == 1]
        # df_non_damaging = self.df.loc[self.df[idx_damaging] == 0]
        # print(df_damaging.shape, df_non_damaging.shape)

        # return train, test, idx_X, idx_A, (idx_faith, idx_damaging)
        return train, test, self.df, idx_X, idx_A, idx_damaging, "wiki"

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
    reader = ParserWiki()
    # reader.create_data()
    # reader.data_stats()
    reader.parse_data()

if __name__ == '__main__':
    main()