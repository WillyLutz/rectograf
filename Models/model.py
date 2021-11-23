import pandas as pd
"""

        self.dataset_headers = ['user_id', 'activity', 'timestamp',
                                'x_axis', 'y_axis', 'z_axis']

    def get_dataset_headers(self):
        return self.dataset_headers

    def set_dataset_headers(self, dataset_headers):
        self.dataset_headers = dataset_headers"""

class Model:
    def __init__(self):
        self.measures = []
        self.dataset = pd.DataFrame()
        self.dataset_path = ""
        self.dataset_filename = ""
        self.current_measure_count = 0
    
    def get_current_measure_count(self):
        return self.current_measure_count

    def set_current_measure_count(self, current_measure_count):
        self.current_measure_count = current_measure_count
    
    def get_dataset_filename(self):
        return self.dataset_filename

    def set_dataset_filename(self, dataset_filename):
        self.dataset_filename = dataset_filename
    
    def get_dataset_path(self):
        return self.dataset_path

    def set_dataset_path(self, dataset_path):
        self.dataset_path = dataset_path

    def get_dataset(self):
        return self.dataset

    def set_dataset(self, dataset):
        self.dataset = dataset

    def get_measures(self):
        return self.measures

    def set_measures(self, measure):
        self.measures = measure
