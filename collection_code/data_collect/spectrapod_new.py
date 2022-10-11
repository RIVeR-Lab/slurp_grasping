#!/usr/bin/python3
import re
import serial
import numpy as np
from copy import copy


# MantiSpectra ROS Driver
# Author: Nathaniel Hanson
# Date: 07/09/2022
# Purpose: ROS Node for interfacing with MantiSpectra Device


class SpectrapodDriver():
    def __init__(self, port_path, int_time_ms):
        # Port name
        self.port_path = port_path
        self.integration_time = int_time_ms/1000

        # # Initialize the spectrometer the Baudrate must equal 115200
        self.spectrometer = serial.Serial(self.port_path, baudrate=115200)
        self.flush()
        # # self.spectrometer.write(self.start_light_power.encode())
        # # self.spectrometer.write('ffv'.encode())
        self.spectrometer.write(b'fo')
        print('spectrapod initialized')

    def flush(self):
        self.spectrometer.flushInput()
        self.spectrometer.flushOutput()


    def process_data(self, data: str) -> None:
        '''
        Take raw data from serial output and parse it into correct format
        '''
        # Remove ANSI escape charters \x1b[36m (cyan)
        ansi_escape = re.compile(r'(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]')
        # print(repr(data))

        # Remove ANSI escape characters
        ansi_removed = ansi_escape.sub('', data)
        # print(ansi_removed)

        # Parse data into array
        parsed_data = np.array(ansi_removed.split(), dtype=np.int32)

        if len(parsed_data) != 20:
            print('Incomplete spectral data received!')
            return None
        else:
            print(len(parsed_data))
            return parsed_data

    def capture_sample(self) -> None:
        '''
        Main operating loop used to read and send data from the spectrometer
        '''
        while True:
            # time.sleep(self.integration_time)
            try:
                # print('grab data')
                # Grab the raw data
                raw_data = self.spectrometer.read_until(b'\r')
                # Decode the spectral data
                spectra_data = raw_data.decode('utf-8')
                # Process and publish the data
                output = self.process_data(spectra_data)
                if output is not None:
                    return output

            except Exception as e:
                pass


    def shutdown(self):
        '''
        Custom shutdown behavior
        '''
        # Turn off the light
        self.spectrometer.write('0v'.encode())
        # Close the serial transmission
        self.spectrometer.close()


# Main functionality
if __name__ == '__main__':
    # Initialize the node and name it.
    
    controller = SpectrapodDriver(port_path = '/dev/ttyUSB2', int_time_ms=512)

    while True:
        samp = controller.capture_sample()
        print(samp)