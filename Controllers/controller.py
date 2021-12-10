from Views.mainView import View
from Models.model import Model
from Controllers.bt_thread import BluetoothServer as BTS
import tkinter as tk
from tkinter import filedialog
import pandas as pd
import os
from pathlib import Path
import io
from functools import partial
import bluetooth
import ast

import subprocess as sp
import Controllers.AiTraining as ai

class Controller:
    def __init__(self, model, view):
        self.view = view
        self.model = model

    def train_model(self):
        df_path = self.model.get_dataset_path()
        df = ai.preprocessing_data(df_path, "RFC")
        if isinstance(df, str):
            print("invalid csv file")
        else:
            ai.random_forest_classifier(df)

    def generate_dataset(self):
        headers = ["x_accel", "y_accel", "z_accel", "impedance", "gesture"]
        name_test = "name_test.csv"
        measurement_items = self.view.measurement_list.get('@1,0', tk.END)
        item_list = list(measurement_items)

        data = pd.DataFrame(columns=["x_accel", "y_accel", "z_accel", "impedance", "gesture"])

        for i in range(len(measurement_items)):
            same_gesture_x = []
            same_gesture_y = []
            same_gesture_z = []
            same_gesture_imp = []
            gesture_label = measurement_items[i][len(measurement_items[i])-1].decode('utf-8')
            for j in range(len(measurement_items[i])-1):
                item = measurement_items[i][j]
                item_content = ast.literal_eval(item.decode('utf-8'))
                accelerations = ast.literal_eval(str(item_content[0]))
                impedance = ast.literal_eval(str(item_content[1]))

                same_gesture_x.append(accelerations[0])
                same_gesture_y.append(accelerations[1])
                same_gesture_z.append(accelerations[2])
                same_gesture_imp.append(impedance)

            row = pd.Series({"x_accel": same_gesture_x, "y_accel": same_gesture_y, "z_accel": same_gesture_z,
                             "impedance": same_gesture_imp, "gesture": gesture_label}, name=i)
            data = data.append(row)
        try:
            with filedialog.asksaveasfile(mode='w', defaultextension=".csv") as file:
                data.to_csv(file.name, index=False)
        except AttributeError:
            pass


    def update_connected_device(self):
        name = self.model.get_connected_device_name()
        addr = self.model.get_connected_device_addr()
        self.view.connected_device_strvar.set(f"Connected device:\n{name}\n{addr}")

    def start_measure(self):
        self.model.clear_received_current_data()
        self.model.set_is_recording(True)

    def connect_to_address(self, addr, name):
        # TODO: effective connection to address
        server_BT = BTS(1, self.model, self, name, addr)
        server_BT.daemon = True
        server_BT.start()

    def connect_to_device(self):
        for i in self.view.addrs_lstbx.curselection():
            fullname = self.view.addrs_lstbx.get(i)
            addr = fullname.split("---")[1]
            name = fullname.split("---")[0]
            self.connect_to_address(addr, name)

    def lookup_devices(self):
        self.view.addrs_lstbx.delete(0, tk.END)
        nearby_devices = bluetooth.discover_devices()

        for bdaddr in nearby_devices:
            name = bluetooth.lookup_name(bdaddr)
            if name is None:
                name = "None"
            device = name + "---" + bdaddr
            self.view.addrs_lstbx.insert(0, device)

    def increase_current_measure_count(self, increment):
        self.model.set_current_measure_count(self.model.get_current_measure_count() + increment)

    def decrease_current_measure_count(self, decrement):
        self.model.set_current_measure_count(self.model.get_current_measure_count() - decrement)

    def from_current_to_overall_data(self):
        overall = self.model.get_received_overall_data()
        current = self.model.get_received_current_data()
        overall.append(current)
        self.model.clear_received_current_data()
        self.model.set_received_overall_data(overall)

    def stop_measure(self):

        if self.model.is_recording:
            self.model.set_is_recording(False)

            entry = self.view.labeled_data_entry.get()
            self.model.set_current_label(entry)
            self.model.del_pending_received_current_data()
            measure = self.model.get_received_current_data()
            if measure:
                entry = [bytes(entry, 'utf-8'), ]
                measure = measure + entry

                self.view.measurement_list.insert(0, measure)
                self.model.clear_received_current_data()

                self.increase_current_measure_count(1)
                self.view.measures_count_strvar.set("Total number of measures : " + str(self.model.get_current_measure_count()))
        else:
            pass

    def remove_measure(self):
        if self.view.measurement_list.curselection():
            self.view.measurement_list.delete(self.view.measurement_list.curselection())
            self.decrease_current_measure_count(1)
            self.view.measures_count_strvar.set("Total number of measures : "
                                                + str(self.model.get_current_measure_count()))
        else:
            self.view.measurement_list.delete(0)
            self.decrease_current_measure_count(1)
            self.view.measures_count_strvar.set("Total number of measures : "
                                                + str(self.model.get_current_measure_count()))

    def load_dataset(self):
        path = os.path.join(Path(os.getcwd()).parent, Path("Models/DATA"))
        dataset_path = filedialog.askopenfilename(initialdir=path, title="Open file")

        self.model.set_dataset_filename(os.path.split(dataset_path)[1])
        self.model.set_dataset_path(dataset_path)

        self.view.loaded_dataset_strvar.set("Loaded dataset :\n " + self.model.get_dataset_filename())

    def display_dataset(self):
        displayed_dataset = tk.Tk()
        displayed_dataset.geometry("350x250")

        dataset_path = self.model.get_dataset_path()
        df = pd.read_csv(dataset_path)

        buf = io.StringIO()
        df.info(buf=buf)
        string_info = buf.getvalue()

        info_df = tk.Text(displayed_dataset)
        info_df.insert(tk.END, string_info)

        info_df.grid(row=1, column=0)

        displayed_dataset.mainloop()

