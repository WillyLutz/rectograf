import bluetooth, subprocess
import pandas as pd
import ast
import Controllers.AiTraining as ai


def get_bt_addresses(target_name):
    huawei_addr = "4C:D1:A1:32:A0:36"
    target_address = None

    nearby_devices = bluetooth.discover_devices()
    for bdaddr in nearby_devices:
        print(bdaddr, " - ", bluetooth.lookup_name(bdaddr))

        if target_name == bluetooth.lookup_name(bdaddr):
            target_address = bdaddr
            break

    if target_address is not None:
        print("found target bluetooth device with address ", target_address)
        return target_address
    else:
        print("could not find bluetooth device")


def rfcomm_server():
    server_sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)

    port = 1
    raspi_addr = "B8:27:EB:8F:BC:78"
    server_sock.bind(("", port))
    print("bind done")

    server_sock.listen(1)
    print("listen done")

    client_sock, address = server_sock.accept()
    print("Accepted connection from ", address)

    data = client_sock.recv(1024)
    print("received [%s]" % data)

    client_sock.close()
    server_sock.close()


pd.set_option('display.max_columns', 5)
path = "../DATA/test_data.csv"
df = pd.read_csv(path)
df["x_accel"] = pd.eval(df["x_accel"])
df["y_accel"] = pd.eval(df["y_accel"])
df["z_accel"] = pd.eval(df["z_accel"])
df["impedance"] = pd.eval(df["impedance"])

print(df)
