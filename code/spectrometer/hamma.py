#!/usr/bin/env python3 
import serial
import csv
import signal
import re
import pandas as pd
import time
import plot_spectrometers
import table_fields
def hamma_take_measurements(port):
    hamma = serial.Serial(port, baudrate=115200, timeout=0.1)
    hamma.flushInput()
    hamma.flushOutput()
    time.sleep(4)

    total = []

    while len(total) != table_fields.MEASUREMENTS:
        time.sleep(1)
        data = hamma.readline()
        decoded_data = data.decode()
        listed_data = decoded_data.split(',')

        if len(listed_data) == 289:
            del listed_data[288]
            listed_data = [int(z) for z in listed_data]
            total.append(listed_data)

    print("SIZE: ",len(total))
    hamma.close()

    return total

hamma_values = hamma_take_measurements('/dev/ttyACM1')
df = pd.DataFrame(hamma_values, columns=table_fields.HAMMAMATSU_FIELDS)

df.to_csv('/home/nathaniel/Downloads/SLURP_ICRA_2023/calibration/hamamatsu_white_ref.csv', index=False)

print(df)
plot_spectrometers.plot_hamma(df, table_fields.HAMMAMATSU_FIELDS, 'TEST')

