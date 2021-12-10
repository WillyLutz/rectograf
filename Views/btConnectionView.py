import tkinter as tk
import tkinter.ttk
from functools import partial


class BtView:

    def __init__(self):
        super().__init__()

        self.connection_win = tk.Tk()
        self.connection_win.geometry("")


        self.xbt_defile_bar = tk.Scrollbar(self.connection_win, orient="horizontal")
        self.ybt_defile_bar = tk.Scrollbar(self.connection_win, orient="vertical")

        self.addrs_lstbx = tk.Listbox(self.connection_win, height=15, width=35, xscrollcommand=self.xbt_defile_bar.set,
                                 yscrollcommand=self.ybt_defile_bar.set)
        self.xbt_defile_bar['command'] = self.addrs_lstbx.xview
        self.ybt_defile_bar["command"] = self.addrs_lstbx.yview

        self.connect_btn = tk.Button(self.connection_win, text="Connect", command=self.connect_device)
        self.lookup_btn = tk.Button(self.connection_win, text="Look up devices", command=self.lookup_devices)

        """-------------- Grid section --------------"""
        self.xbt_defile_bar.grid(row=1, column=0, columnspan=2)
        self.ybt_defile_bar.grid(row=2, column=1)
        self.addrs_lstbx.grid(row=2, column=0)
        self.lookup_btn.grid(row=3, column=0)
        self.connect_btn.grid(row=3, column=1)

        self.controller = None


    def lookup_devices(self):
        if self.controller:
            self.controller.lookup_devices()

    def connect_device(self):
        if self.controller:
            self.controller.connect_to_device()

    def set_controller(self, controller):
        self.controller = controller
