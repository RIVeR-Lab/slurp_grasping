#!/usr/bin/env python3
import serial
import csv
import signal
import re
import time
import plot_spectrometers
import table_fields
import pandas as pd
# For the best performance keep the terminal large enough to display all 20 values. There is data loss when the raw data wraps

# Enter name of the csv file
print("File Name: ")
file_name = input() + '.csv'

# Port names
spectrapod_port_path = '/dev/ttyUSB0'
hammamatsu_port_path = '/dev/ttyACM0' # using an integration value of 85 6v


# ENTER YOUR OWN PATH
file_path = '/home/knaber/spectromeater/data/' + file_name

# Initialize the Hammamatsu
# The Baudrate must equal 115200
hammamatsu = serial.Serial(hammamatsu_port_path, baudrate=115200, timeout=0.1)
hammamatsu.flushInput()
hammamatsu.flushOutput()

# Initialize the Spectrapod
# The Baudrate must equal 115200
spectrapod = serial.Serial(spectrapod_port_path, baudrate=115200, timeout=0.1)
spectrapod.write(b'fo')
spectrapod.flushInput()
spectrapod.flushOutput()

# Open temporary file to write to
file = open(file_path, 'w')

def cleaned_data():
    # Input the processed data after gathering data from the Spectrapod
    processed_file = open(file_path, "r")
    file_lines = processed_file.readlines()

    # Put array of size 20 into a temporary array called full_sized_arrays
    # ideally all data should be 20 values long
    full_sized_arrays = []
    for i in range(len(file_lines)):
        y = file_lines[i].split()
        if len(y) == 20:
            full_sized_arrays.append(y)

    # Remove the arrays that contain letters
    # First add all the arrays with letters in remove_letters_list
    remove_letters_list = []
    for i in range(len(full_sized_arrays)):
        for j in full_sized_arrays[i]:
            if not re.match(r"[-+]?(?:\d*\.\d+|\d+)", j):
                remove_letters_list.append(full_sized_arrays[i])

    # Secondly, remove it from the full_sized_arrays
    for i in range(len(remove_letters_list)):
        full_sized_arrays.remove(remove_letters_list[i])

    # Shrink down the arrays to the amount of measurements needed
    final_table = []
    for i in range(len(full_sized_arrays)):
        if (table_fields.MEASUREMENTS > i):
            final_table.append(full_sized_arrays[i])

    return final_table


# Remove ANSI escape charters \x1b[36m (cyan)
ansi_escape = re.compile(r'(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]')

# Grab the numbers
just_numbers = re.compile(r"[-+]?(?:\d*\.\d+|\d+)")

# (16 channels + 4 parameters) * 5 readings = 100 data points per object
# param_1: OSR, equivalent to integration time: a value ranging from 1 (high speed) to 15 (low speed)
# param_2: Internal Lamp power setting: 0 = not activated, 255 = full lamp power
# param_3: Humidity sensor
# param_4: Temperature sensor
hamma_total = []
for i in range(table_fields.MEASUREMENTS):

    if i != 0 and (i)%2==0:
        print()
        print("Change the orientation!, {}".format(i))
        change = input()

    spectrapod.flushInput()
    spectrapod.flushOutput()

    hammamatsu.flushInput()
    hammamatsu.flushOutput()

    time.sleep(4.5)  # This might seem long but the arrays wont come fully otherwise

    spectra_data = spectrapod.readline()
    # Decode the raw data
    decoded_data = spectra_data.decode('utf-8')
    # Remove ANSI escape characters
    ansi_removed = ansi_escape.sub('', decoded_data)
    print(ansi_removed)
    print()
    while len(hamma_total) != i+1:
        hama_data = hammamatsu.readline()
        hama_decoded_data = hama_data.decode()
        hama_listed_data = hama_decoded_data.split(',')

        if len(hama_listed_data) == 289:
            del hama_listed_data[288]
            hamma_total.append(hama_listed_data)
            print(hama_listed_data)

    # Write to csv file
    file.write(ansi_removed + '\n')


print("HAMAMATSU ARRAY SIZE: ",len(hamma_total))


# Close the hammamatsu port
hammamatsu.close()
# Close the spectrapod port
spectrapod.close()

hamma_df = pd.DataFrame(hamma_total, columns=table_fields.HAMMAMATSU_FIELDS)
spec_df = pd.DataFrame()

# Close the file
file.close()

# Clean the data
cleaned_csv = cleaned_data()

time.sleep(1)
# Display a table of the final table
def display_table(path, final_table):
    results = []
    with open(path, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)

        # writing the fields
        csvwriter.writerow(table_fields.SPECTRAPOD_FIELDS)

        # over-writing the data rows
        csvwriter.writerows(final_table)

    df = pd.read_csv(path)
    plot_spectrometers.plot_spectrapod(df, table_fields.SPECTRAPOD_PLOT_FIELDS, file_name)
    for i in range(table_fields.MEASUREMENTS):
        first = df.iloc[i, :].values.tolist()
        second = hamma_df.iloc[i, :].values.tolist()
        second.extend(first)
        results.append(second)

    df = pd.DataFrame(results, columns=table_fields.ALL_FIELDS)
    df["Container&Substrate"] = str(file_name.replace('.csv', ''))
    df.to_csv(path, index=False)
    print(df)



# Display Table
display_table(file_path, cleaned_csv)

plot_spectrometers.plot_hamma(hamma_df, table_fields.HAMMAMATSU_FIELDS, file_name)

