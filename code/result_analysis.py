import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

class Analysis:
    def __init__(self):
        self.file = 'dataset/analysis/raw_data.csv'

        self.data = pd.read_csv(self.file)
        print(self.data.head(3))
        self.data = self.data.drop(self.data.index[0])
        print("############################")
        print(self.data.head(3))

    def data_cleaning(self):
        # remove those failed attention check
        print(self.data.shape)
        self.data = self.data[self.data.SC1 == '1']  # remove 5 data points ...
        print(self.data.shape)

        print(self.data.columns.values)

        # time unit conversion
        self.data['Duration (in seconds)'] = self.data['Duration (in seconds)'].astype('int32')
        self.data['Duration (in seconds)'] /= 60

        self.data['Q35'] = self.data['Q35'].astype('int32')
        self.data['Q50'] = self.data['Q50'].astype('int32')
        self.data['Q51'] = self.data['Q51'].astype('int32')
        self.data['Q53'] = self.data['Q53'].astype('int32')
        self.data['Q54'] = self.data['Q54'].astype('int32')
        self.data['Q55'] = self.data['Q55'].astype('int32')
        self.data['Q59'] = self.data['Q59'].astype('int32')
        self.data['Q62'] = self.data['Q62'].astype('int32')
        self.data['Q64'] = self.data['Q64'].astype('int32')
        self.data['Q65'] = self.data['Q65'].astype('int32')
        self.data['SC0'] = self.data['SC0'].astype('int32')

        self.data = self.data.rename(columns={"Q50": "Confident in responses", "Q51": "Help understand trade-offs",
                                              "Q53": "Familiar with the tool quickly", "Q54": "Ease of use", "Q59": "Age",
                                              "Q62": "Education level", "Q63": "Race", "SC0": "Score",
                                              "Q64": "Familiarity with judicial system",
                                              "Q65": "Familiarity with AI-powered systems",
                                              "Duration (in seconds)": "Duration (in minutes)"})
        self.data['Trust change'] = self.data['Q55'] - self.data['Q35']

        # create dummy variables
        df_dummies = pd.get_dummies(self.data.condition.astype('category'))
        self.data = pd.concat([self.data, df_dummies], axis=1)

        print(self.data.shape)
        print(self.data.columns.values)

    def run(self):
        self.data_cleaning()
        df = self.data
        self.correlation_analysis(df)

        feature = ['Age', 'Education level',
                   'Familiarity with judicial system', 'Familiarity with AI-powered systems', 'data', 'scenario']
        print(df[feature].corr(method='pearson'))

        # IVs: age, education,
        # TODO: add in race and gender
        # age, education, familiarity with judicial system, Your familiarity of the use of AI-powered systems, dummy
        X = df.loc[:, ['Age', 'Education level', 'Familiarity with judicial system',
                       'Familiarity with AI-powered systems', 'data', 'scenario']]

        # DV1: score
        # DVs for different models: trust, time, confidence, evaluation results
        print('***DV: objective evaluation score')
        y = df.loc[:, df.columns == 'Score']
        est = sm.OLS(y, sm.add_constant(X))
        est2 = est.fit()
        print(est2.summary())

        # DV2: self eval (self confidence)
        print('***DV: self confidence')
        y = df.loc[:, df.columns == 'Confident in responses']
        est = sm.OLS(y, sm.add_constant(X))
        est2 = est.fit()
        print(est2.summary())

        # DV3: self eval on understanding trade-offs
        print('***DV: understanding trade-offs')
        y = df.loc[:, df.columns == 'Help understand trade-offs']
        est = sm.OLS(y, sm.add_constant(X))
        est2 = est.fit()
        print(est2.summary())

        # DV4: trust change
        print('***DV: Trust change')
        y = df.loc[:, df.columns == 'Trust change']
        est = sm.OLS(y, sm.add_constant(X))
        est2 = est.fit()
        print(est2.summary())

        # DV5: finishing time
        print('***DV: finishing time in minutes')
        y = df.loc[:, df.columns == 'Duration (in minutes)']
        est = sm.OLS(y, sm.add_constant(X))
        est2 = est.fit()
        print(est2.summary())

        # DV5: ease of use
        print('***DV: ease of use')
        y = df.loc[:, df.columns == 'Ease of use']
        est = sm.OLS(y, sm.add_constant(X))
        est2 = est.fit()
        print(est2.summary())

    def corr_mtx(self, df, dropDuplicates=True):

        # Compute the correlation matrix
        df = df.corr()

        # Exclude duplicate correlations by masking uper right values
        if dropDuplicates:
            mask = np.zeros_like(df, dtype=np.bool)
            mask[np.triu_indices_from(mask)] = True

        # Set background color / chart style
        sns.set_style(style='white')

        # Set up  matplotlib figure
        f, ax = plt.subplots(figsize=(11, 9))

        # Add diverging colormap from red to blue
        cmap = sns.diverging_palette(250, 10, as_cmap=True)

        # Draw correlation plot with or without duplicates
        if dropDuplicates:
            sns.heatmap(df, mask=mask, cmap=cmap, square=True, linewidth=.5, cbar_kws={"shrink": .5}, ax=ax)
        else:
            sns.heatmap(df, cmap=cmap, square=True, linewidth=.5, cbar_kws={"shrink": .5}, ax=ax)
        plt.xticks(rotation=45)
        plt.show()

    def correlation_analysis(self, data):
        # df = pd.read_csv(self.const.f_data_regression, delimiter=',')
        # print(df.head(5))

        feature = ['Age', 'Education level',
                   'Familiarity with judicial system', 'Familiarity with AI-powered systems', 'data', 'scenario']
        self.corr_mtx(data[feature], True)

def main():
    Analysis().run()


if __name__ == '__main__':
    main()