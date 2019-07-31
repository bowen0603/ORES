import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

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
        self.data['Q70'] = self.data['Q70'].astype('int32')
        self.data['Q72'] = self.data['Q72'].astype('int32')
        self.data['SC0'] = self.data['SC0'].astype('int32')

        # self.data = self.data.replace({'condition': {'data': 0, 'scenario': 1}})
        #
        # self.data = self.data.replace({'Q63': {'1': 'White', '2': 'African',
        #                                         '3': 'Hispanic', '4': 'Asian', '5': 'Others'}})
        # df_dummies = pd.get_dummies(self.data.Q63.astype('category'))
        #
        # self.data = self.data.replace({'Q61': {'female': 'Female', 'male': 'Male'}})
        # self.data = self.data.replace({'Q61': {'Female': 0, 'Male': 1}})
        # self.data['Q61'] = self.data['Q61'].astype('int32')
        # self.data = pd.concat([self.data, df_dummies], axis=1)
        # print(self.data.loc[:, ['Q59', 'Q61', 'Q62', 'Q64', 'Q65', 'condition',
        # 'White', 'African', 'Hispanic', 'Asian', 'Others']].corr())

        self.data = self.data.rename(columns={"Q50": "Confident in responses", "Q51": "Help understand trade-offs",
                                              "Q53": "Familiar with the tool quickly", "Q54": "Ease of use",
                                              "Q59": "Age", "Q61": "Gender",
                                              "Q62": "Education level", "Q63": "Race", "SC0": "Score",
                                              "Q64": "Familiarity with judicial system",
                                              "Q65": "Familiarity with AI-powered systems",
                                              "Q70": "Reflect value about disparity-error",
                                              "Q72": "Reflect value about aggressiveness",
                                              "Duration (in seconds)": "Duration (in minutes)"})
        self.data['Trust change'] = self.data['Q55'] - self.data['Q35']
        print(self.data.columns.values)

        # TODO: convert condition to 0/1
        # self.data = self.data.replace({'condition': {'data': 0, 'scenario': 1}})
        # self.data = self.data.rename(columns={'condition': 'isScenario'})

        # convert gender to 0/1
        self.data = self.data.replace({'Gender': {'female': 'Female', 'male': 'Male'}})
        self.data = self.data.replace({'Gender': {'Female': 0, 'Male': 1}})
        self.data['Gender'] = self.data['Gender'].astype('int32')

        # convert race to categorical (dummy)
        self.data = self.data.replace({'Race': {'1': 'White', '2': 'African',
                                                '3': 'Hispanic', '4': 'Asian', '5': 'Others'}})
        df_dummies = pd.get_dummies(self.data.Race.astype('category'))
        self.data = pd.concat([self.data, df_dummies], axis=1)
        self.data = self.data.drop(columns='Others')  # for dummy variables

    def analysis_plots(self, df):
        DVs = ['Score', 'Confident in responses', 'Help understand trade-offs', 'Trust change',
                    'Duration (in minutes)', 'Ease of use', 'Reflect value about aggressiveness',
                    'Reflect value about disparity-error']
        for dv in DVs:
            # Reflect value about aggressiveness
            df_cond = df[['condition', dv]]
            df_cond = df_cond.groupby(['condition', dv]).size().reset_index(name='counts')
            df_cond.pivot(index=dv, columns='condition', values='counts').plot(kind='bar')
            plt.legend(loc='best')
            plt.title('{} distribution'.format(dv))
            plt.xticks(rotation=45)
            plt.show()

        IVs = ['Age', 'Education level', 'Familiarity with judicial system', 'Gender',
                       'Familiarity with AI-powered systems']
        for iv in IVs:
            # Reflect value about aggressiveness
            df_cond = df[['condition', iv]]
            df_cond = df_cond.groupby(['condition', iv]).size().reset_index(name='counts')
            df_cond.pivot(index=iv, columns='condition', values='counts').plot(kind='bar')
            plt.legend(loc='best')
            plt.title('{} distribution'.format(iv))
            plt.xticks(rotation=45)
            plt.show()

    def analysis_correct_questions(self, df):
        # select questions
        list_questions = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q41', 'Q42', 'Q43', 'Q44', 'Q45', 'Q9', 'Q11']
        list_answers = ['1', '1', '2', '1', '2', '1', '1', '1', '2', '2', '1', '1']

        print('Question, data, scenario')
        for idx in range(len(list_questions)):
            question = list_questions[idx]
            answer = list_answers[idx]

            df_cond = df[['condition', question]]
            df_cond = df_cond.groupby(['condition', question]).size().reset_index(name='counts')

            d_data_corr = df_cond.loc[(df_cond['condition'] == 'data') & (df_cond[question] == answer)]
            d_scen_corr = df_cond.loc[(df_cond['condition'] == 'scenario') & (df_cond[question] == answer)]

            print('{}\t{}\t{}'.format(question, round(d_data_corr['counts'].values[0]/52, 2),
                                                 round(d_scen_corr['counts'].values[0]/45, 2)))


    def run_regression(self, df):
        # IVs: age, education, familiarity with judicial system, Your familiarity of the use of AI-powered systems
        X = df.loc[:, ['Age', 'Education level', 'Familiarity with judicial system', 'Gender',
                       'Familiarity with AI-powered systems', 'isScenario', 'White', 'African', 'Hispanic', 'Asian']]

        df_data_view = df[df['isScenario'] == 0]
        df_scenario_view = df[df['isScenario'] == 1]

        list_DVs = ['Score', 'Confident in responses', 'Help understand trade-offs', 'Trust change',
                    'Duration (in minutes)', 'Ease of use', 'Reflect value about aggressiveness',
                    'Reflect value about disparity-error']

        for DV in list_DVs:
            print('***DV: {}'.format(DV))
            y = df.loc[:, df.columns == DV]
            est = sm.OLS(y, sm.add_constant(X))
            est2 = est.fit()
            print(est2.summary())
            # t, p = ttest_ind(df_data_view.loc[:, DV].tolist(),
            #                  df_scenario_view.loc[:, DV].tolist(), equal_var=False)
            # print('t-test: t: {}, pval: {}'.format(t, p))

    def run(self):
        self.data_cleaning()
        df = self.data

        self.analysis_correct_questions(df)
        # self.analysis_plots(df)
        # self.run_regression(df)
        # self.correlation_analysis(df)

        # feature = ['Age', 'Education level', 'Familiarity with judicial system', 'Gender',
        #            'Familiarity with AI-powered systems', 'condition']
        # print(df[feature].corr(method='pearson'))


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
                   'Familiarity with judicial system', 'Familiarity with AI-powered systems', 'isScenario', 'Gender']
        self.corr_mtx(data[feature], True)

def main():
    Analysis().run()


if __name__ == '__main__':
    main()