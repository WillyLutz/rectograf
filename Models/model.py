import pandas as pd


class Model:
    def __init__(self):
        self.measures = []
        self.dataset = pd.DataFrame()
        self.dataset_path = ""
        self.dataset_filename = ""
        self.current_measure_count = 0
        self.connected_device_addr = ""
        self.connected_device_name = ""
        self.received_current_data = []
        self.received_overall_data = []
        self.current_label = ""
        self.is_recording = False

    def get_current_label(self):
        return self.current_label

    def set_current_label(self, label):
        self.current_label = label

    def get_is_recording(self):
        return self.is_recording

    def set_is_recording(self, state):
        self.is_recording = state

    def clear_received_current_data(self):
        self.received_current_data.clear()

    def del_pending_received_current_data(self):
        if self.received_current_data:
            self.received_current_data.pop(0)

    def get_received_overall_data(self):
        return self.received_overall_data

    def append_received_overall_data(self, overall_data):
        self.received_overall_data.append(overall_data)

    def set_received_overall_data(self, overall_data):
        self.received_overall_data = overall_data
    
    def get_received_current_data(self):
        return self.received_current_data

    def append_received_current_data(self, current_data):
        self.received_current_data.append(current_data)

    def set_received_current_data(self, current_data):
        self.received_current_data = current_data
    
    def get_connected_device_name(self):
        return self.connected_device_name

    def set_connected_device_name(self, measure):
        self.connected_device_name = measure
    
    def get_connected_device_addr(self):
        return self.connected_device_addr

    def set_connected_device_addr(self, measure):
        self.connected_device_addr = measure
    
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
