#!/usr/bin/python3
import re
import os
import csv
import time
import serial
import numpy as np

# Hamamatsu Spectrometer ROS Driver
# Author: Nathaniel Hanson
# Date: 07/09/2022
# Purpose: ROS Node for interfacing with MantiSpectra Device

class HamamatsuDriver():
    def __init__(self, port_path, int_time_ms):
        # Initialize empty store to aggregate spectral readings
        self.store = []
        self.last = []
        self.collection_on = False
        self.msg_count = 0
        # Buffer of commands to write to the device
        self.command_buffer = []

        self.port_path = port_path
        self.integration_time = int_time_ms/1000

        # # Grab parameters from the ros param server
        # # Integration time in seconds
        # self.integration_time = rospy.get_param('integration_time', 0.100)
        # self.start_light_power = rospy.get_param('light_power', 0)
        # self.white_ref = self.load_calibration(rospy.get_param('white_cal', os.path.join(rospack.get_path('spectrometer_drivers'),'data','hamamatsu_white_ref.txt')))
        # self.dark_ref = self.load_calibration(rospy.get_param('dark_cal' ,os.path.join(rospack.get_path('spectrometer_drivers'),'data', 'hamamatsu_dark_ref.txt')))
        # self.wavelengths = rospy.get_param('device_wavelengths', None)
        # if self.wavelengths == None:
        self.wavelengths = list(np.linspace(340,850,288))
        # # Port name
        # self.port_path = rospy.get_param('port', '/dev/ttyACM1')

        # Initialize the spectrometer the Baudrate must equal 115200
        self.spectrometer = serial.Serial(self.port_path, baudrate=115200)

        self.flush()

        # # Initialize publishers
        # self.pub = rospy.Publisher('/hamamatsu/data', Spectra, queue_size=10)
        # self.pub_cal = rospy.Publisher('/hamamatsu/data_cal', Spectra, queue_size=10)
        
        # # Initialize collection services
        # self.service_start = rospy.Service('/hamamatsu/request_start', StartCollect, self.start_collect)
        # self.service_end = rospy.Service('/hamamatsu/request_end', EndCollect, self.end_collect)
        # self.service_once = rospy.Service('/hamamatsu/request_sample', RequestOnce, self.realtime_read)
        
        # # Initialize serivce utilities
        # self.service_light = rospy.Service("/hamamatsu/light_power", Light, self.set_light_power)
        # self.service_integration = rospy.Service("/hamamatsu/integration_time", Integration, self.set_integration_time)
        # # Define shutdown behavior
        # rospy.on_shutdown(self.shutdown)
        print('hamamatsu initialized')
    
    def flush(self):
        self.spectrometer.flushInput()
        self.spectrometer.flushOutput()


    def process_data(self, data: str) -> None:
        '''
        Take raw data from serial output and parse it into correct format
        '''
        # Remove ANSI escape charters \x1b[36m (cyan)
        ansi_escape =re.compile(r'(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]')

        # Remove ANSI escape characters
        ansi_removed = ansi_escape.sub('', data)
        # Parse data into array
        parsed_data = np.array([int(z.strip()) for z in ansi_removed.split(',')[:-1]], dtype=np.int32)
        # print(f'Data length: {len(parsed_data)} {len(self.wavelengths)}')
        if len(parsed_data) != len(self.wavelengths):
            print('Incomplete spectral data received!')
            return None
        else:
            print(len(parsed_data))
            # print(parsed_data)
            return parsed_data

    def capture_sample(self) -> None:
        '''
        Main operating loop used to read and send data from the spectrometer
        '''
        while True:
            try:
                # Grab the raw data
                raw_data = self.spectrometer.readline()
                # Decode the spectral data
                spectra_data = raw_data.decode('utf-8').strip()
                # Process and publish the data
                output = self.process_data(spectra_data)
                if output is not None:
                    return output
                # Write the latest command
                # self.write_commands()
            except Exception as e:
                pass
            time.sleep(self.integration_time)

    def shutdown(self):
        '''
        Custom shutdown behavior
        '''
        # Turn off the light
        self.spectrometer.write('light_power OFF'.encode())
        # Close the serial transmission
        self.spectrometer.close()

# Main functionality
if __name__ == '__main__':
    # Initialize the node and name it.
    controller = HamamatsuDriver(port_path='/dev/ttyACM6', int_time_ms=512)

    while True:
        samp = controller.capture_sample()
        print(samp)