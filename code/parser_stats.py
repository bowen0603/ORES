__author__ = 'bobo'

from parser_adult import Adult
from parser_dutch import Dutch
from parser_lawschool import Lawschool
from parser_campus import Campus
from parser_wiki import ParserWiki

class PaserStats:
    def __init__(self):
        pass

    # 37155 v.s. 11687
    @staticmethod
    def stats_data_adult():
        print("Adult data")
        adult = Adult()
        data = adult.load_data()


        print(data['label'].nunique(), data[data.label == 0].shape, data[data.label == 1].shape)

    @staticmethod
    # 5099 v.s. 2819
    def stats_data_campus():
        print("\nCampus data")
        campus = Campus()
        data = campus.load_data()

        print(data['label'].nunique(), data[data.label == 0].shape, data[data.label == 1].shape)

    @staticmethod
    # 947 v.s. 19702
    def stats_data_lawschool():
        print("\nLaw school data")
        law = Lawschool()
        data = law.load_data()

        print(data['label'].nunique(), data[data.label == 0].shape, data[data.label == 1].shape)

    @staticmethod
    # 31657 v.s. 28763
    def stats_data_dutch():
        print("\nDutch census data")
        dutch = Dutch()
        data = dutch.load_data()

        print(data['label'].nunique(), data[data.label == 0].shape, data[data.label == 1].shape)

    @staticmethod
    # 18661 v.s.751
    # 15924 v.s. 3488
    def stats_data_wiki():
        print("\nWikipedia data")
        wiki = ParserWiki()
        data = wiki.load_data()

        print(data[1].nunique(), data[data[1] == 0].shape, data[data[1] == 1].shape)
        print(data[2].nunique(), data[data[2] == 0].shape, data[data[2] == 1].shape)

    def run(self):
        # self.stats_data_adult()
        self.stats_data_campus()
        self.stats_data_dutch()
        self.stats_data_lawschool()
        self.stats_data_wiki()


def main():
    PaserStats().run()

if __name__ == '__main__':
    main()