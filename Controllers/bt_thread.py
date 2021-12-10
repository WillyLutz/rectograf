import threading
import time
import bluetooth

class BluetoothServer(threading.Thread):
    def __init__(self, threadID, model, controller, name, addr):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.addr = addr
        self.model = model
        self.controller = controller

    def run(self):
        server_sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
        port = 0
        server_sock.bind(("", port))
        server_sock.listen(1)
        client_sock, address = server_sock.accept()

        self.model.set_connected_device_addr(self.addr)
        self.model.set_connected_device_name(self.name)
        self.controller.update_connected_device()
        while self.is_alive():
            if self.model.is_recording:
                data = client_sock.recv(1024)
                if data:
                    self.model.append_received_current_data(data)
        client_sock.close()
        server_sock.close()