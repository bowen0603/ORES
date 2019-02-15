__author__ = 'bobo'

class FileReader:
    def __init__(self):
        self.data_x = None
        # todo: add protected features (registration, gender)
        # either the first two or the last two columns
        self.data_y_badfaith = None
        self.data_y_damaging = None


    def read_from_file(self):
        # todo: read from files ...
        pass

    def group_identifier(self):
        # revision id -> editor id -> editor page/info
        # todo: registered or not
        # todo: female or male
        # todo: append attribute to x data
        pass

    def data_stats(self):
        # todo: % of registered or not, editors
        # todo: % of female, male, or none editors
        pass