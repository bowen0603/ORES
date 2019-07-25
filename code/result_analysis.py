import pandas as pd
import statsmodels.api as sm

class Analysis:
    def __init__(self):
        self.file = 'dataset/analysis/raw_data.csv'


        self.data = pd.read_csv(self.file)
        print(self.data.head(3))
        print(type(self.data))
        self.data = self.data.drop(self.data.index[0])
        print("############################")
        print(self.data.head(3))
        print(type(self.data))
        # print(self.data.shape)

    def data_cleaning(self):
        # remove those failed attention check

        # time unit converstion
        self.data['Duration (in seconds)'] /= 60

        # create dummy variables
        self.data = pd.get_dummies(self.data.condition.astype('category'))
        print(self.data.head(3))

    def run(self):
        self.data_cleaning()
        df = self.data

        # OLS with different DVs

        X = df.loc[:, df.columns != 'requests']
        y = df.loc[:, df.columns == 'requests']

        est = sm.OLS(y, sm.add_constant(X))
        est2 = est.fit()
        print(est2.summary())

        pass


def main():
    Analysis().run()

if __name__ == '__main__':
    main()