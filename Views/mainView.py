import tkinter as tk
import tkinter.ttk
from functools import partial


class View:

    def __init__(self, app):
        super().__init__()
        self.app = app

        """
        ------------------------------------------------------------------
        |   bar menu(connectDevice, saveFile, openModel)                 |
        |                                                                |
        |   **********************************************************   |
        |   *                                                        *   |
        |   *                                                        *   |
        |   *            Canvas for displaying things                *   |
        |   *                                                        *   |
        |   *                                                        *   |
        |   *                                                        *   |
        |   *                                                        *   |
        |   **********************************************************   |                                             |
        |        In-Use Models                  connected device          |
        |       x defile bar                                             |
        |   *******************                                          |
        |   *                 *                                          |
        |   *                 *                                          |
        |   *   measurement   * y                                       |
        |   *      list       * defile                *start measure*    |
        |   *                 * bar                   *stop measure*     |
        |   *                 *                                          |
        |   *                 *                       *Train model*      |
        |   *******************                       *Test Models*       |
        |                                                                |
        ------------------------------------------------------------------
        """
        """------------ menu bar --------------"""
        self.menu_bar = tk.Menu()
        self.file_menu = tk.Menu(self.menu_bar)
        self.file_menu.add_command(label="Save data")
        self.file_menu.add_command(label="Load data")
        self.file_menu.add_command(label="Exit")

        self.model_menu = tk.Menu(self.menu_bar)
        self.model_menu.add_command(label="Load model")
        self.model_menu.add_command(label="Save model")

        self.display_menu = tk.Menu(self.menu_bar)
        self.display_menu.add_command(label="Dataset")

        self.menu_bar.add_cascade(label="File", menu=self.file_menu)
        self.menu_bar.add_cascade(label="Models", menu=self.model_menu)
        self.menu_bar.add_cascade(label="Display", menu=self.display_menu)

        app.config(menu=self.menu_bar)

        """------------ Widgets ------------------"""
        self.main_canvas = tk.Canvas(bg="white", height=250, width=750)

        self.in_use_model_strvar = tk.StringVar()
        self.in_use_model_strvar.set("In-use AI model :\n None")
        self.in_use_model_label = tk.Label(textvariable=self.in_use_model_strvar)

        self.connected_device_strvar = tk.StringVar()
        self.connected_device_strvar.set("Connected device :\n None")
        self.connected_device_label = tk.Label(textvariable=self.connected_device_strvar)

        self.loaded_dataset_strvar = tk.StringVar()
        self.loaded_dataset_strvar.set("Loaded dataset :\n None")
        self.loaded_dataset = tk.Label(textvariable=self.loaded_dataset_strvar)


        self.x_defile_bar = tk.Scrollbar(orient="horizontal")
        self.y_defile_bar = tk.Scrollbar(orient="vertical")

        self.measurement_list = tk.Listbox(height=15, width=70,
                                           xscrollcommand=self.x_defile_bar.set,
                                           yscrollcommand=self.y_defile_bar.set)
        self.x_defile_bar['command'] = self.measurement_list.xview
        self.y_defile_bar["command"] = self.measurement_list.yview

        self.del_measurement_btn = tk.Button(text="Remove measure")

        self.measures_count_strvar = tk.StringVar()
        self.measures_count_strvar.set("Total number of measures : None")
        self.measures_count = tk.Label(textvariable=self.measures_count_strvar)

        self.labeled_data_entry = tk.Entry(width=20)
        self.labeled_data_entry.insert(0, "Default label")

        self.start_measure_btn = tk.Button(text="Start measure")
        self.stop_measure_btn = tk.Button(text="Stop measure")

        self.train_model_btn = tk.Button(text="Train AI model")
        self.test_model_btn = tk.Button(text="Test AI model")

        self.xbt_defile_bar = tk.Scrollbar(orient="horizontal")
        self.ybt_defile_bar = tk.Scrollbar(orient="vertical")

        self.addrs_lstbx = tk.Listbox(height=15, width=35, xscrollcommand=self.xbt_defile_bar.set,
                                      yscrollcommand=self.ybt_defile_bar.set)
        self.xbt_defile_bar['command'] = self.addrs_lstbx.xview
        self.ybt_defile_bar["command"] = self.addrs_lstbx.yview

        self.connect_btn = tk.Button(text="Connect")
        self.lookup_btn = tk.Button(text="Look up devices")

        self.generate_dataset_btn = tk.Button(text="Generate dataset")

        """---------- Button Commands ----------"""
        self.file_menu.entryconfigure(2, command=self.load_dataset)
        self.file_menu.entryconfigure(3, command=app.quit)

        self.display_menu.entryconfigure(1, command=self.display_dataset)

        self.start_measure_btn.configure(command=self.start_measure)
        self.stop_measure_btn.configure(command=self.stop_measure)
        self.del_measurement_btn.configure(command=self.remove_measure_from_list)

        self.connect_btn.configure(command=self.connect_device)
        self.lookup_btn.configure(command=self.lookup_devices)

        self.generate_dataset_btn.configure(command=self.generate_dataset)

        self.train_model_btn.configure(command=self.train_model)
        """---------- Grid Section ---------"""
        self.main_canvas.grid(row=0, column=0, columnspan=6)

        self.in_use_model_label.grid(row=1, column=0)
        self.loaded_dataset.grid(row=1, column=1)
        self.connected_device_label.grid(row=1, column=2)

        self.measurement_list.grid(row=4, column=0, rowspan=4, columnspan=3)
        self.x_defile_bar.grid(row=4, column=0, sticky='ew', columnspan=3)
        self.y_defile_bar.grid(row=5, column=3, sticky='ns', rowspan=2)

        self.del_measurement_btn.grid(row=9, column=0)
        self.measures_count.grid(row=10, column=0)

        self.labeled_data_entry.grid(row=8, column=0)

        self.start_measure_btn.grid(row=8, column=1)
        self.stop_measure_btn.grid(row=9, column=1)
        self.train_model_btn.grid(row=8, column=2)
        self.test_model_btn.grid(row=9, column=2)

        self.xbt_defile_bar.grid(row=4, column=4, sticky='ew', columnspan=2)
        self.ybt_defile_bar.grid(row=5, column=6, sticky='ns', rowspan=2)
        self.addrs_lstbx.grid(row=5, column=4, columnspan=2)
        self.lookup_btn.grid(row=6, column=4)
        self.connect_btn.grid(row=6, column=5)
        self.generate_dataset_btn.grid(row=10, column=1)

        self.controller = None

    def train_model(self):
        if self.controller:
            self.controller.train_model()

    def generate_dataset(self):
        if self.controller:
            self.controller.generate_dataset()

    def start_measure(self):
        if self.controller:
            self.controller.start_measure()

    def display_dataset(self):
        if self.controller:
            self.controller.display_dataset()

    def load_dataset(self):
        if self.controller:
            self.controller.load_dataset()

    def remove_measure_from_list(self):
        if self.controller:
            self.controller.remove_measure()

    def stop_measure(self):
        if self.controller:
            self.controller.stop_measure()

    def set_controller(self, controller):
        self.controller = controller

    def create_bt_interface(self):
        if self.controller:
            self.controller.create_bt_interface()

    def lookup_devices(self):
        if self.controller:
            self.controller.lookup_devices()

    def connect_device(self):
        if self.controller:
            self.controller.connect_to_device()

