from Views.view import View
from Models.model import Model
import tkinter as tk
from tkinter import filedialog
import pandas as pd
import os
from pathlib import Path
import io


class Controller:
    def __init__(self, model, view):
        self.view = view
        self.model = model

    def increase_current_measure_count(self, increment):
        self.model.set_current_measure_count(self.model.get_current_measure_count() + increment)
        
    def decrease_current_measure_count(self, decrement):
        self.model.set_current_measure_count(self.model.get_current_measure_count() - decrement)
    
    def add_measure(self):
        entry = self.view.labeled_data_entry.get()
        self.view.measurement_list.insert(0, entry)
        self.increase_current_measure_count(1)
        self.view.measures_count_strvar.set("Total number of measures : " + str(self.model.get_current_measure_count()))

    def remove_measure(self):
        if self.view.measurement_list.curselection():
            self.view.measurement_list.delete(self.view.measurement_list.curselection())
            self.decrease_current_measure_count(1)
            self.view.measures_count_strvar.set("Total number of measures : "
                                                + str(self.model.get_current_measure_count()))

    def load_dataset(self):
        path = os.path.join(Path(os.getcwd()).parent, Path("Models/DATA"))
        dataset_path = filedialog.askopenfilename(initialdir=path, title="Open file")

        self.model.set_dataset_filename(os.path.split(dataset_path)[1])
        self.model.set_dataset_path(dataset_path)

        self.view.loaded_dataset_strvar.set("Loaded dataset :\n "+ self.model.get_dataset_filename())

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
