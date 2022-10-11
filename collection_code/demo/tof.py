import serial, time, re

# VL53L4CD ROS Driver
# Author: Nathaniel Hanson
# Date: 08/02/2022
# Purpose: ROS Node for interfacing with VL53L4CD ToF distance sensor

class ToFDriver():
    def __init__(self, port_path):
        # Initialize empty store to aggregate spectral readings
        # Port name
        self.port_path = port_path

        # Initialize the spectrometer the Baudrate must equal 115200
        self.tof = serial.Serial(self.port_path, baudrate=115200)
        self.flush()
        print('ToF initialized')

    def flush(self):
        self.tof.flushInput()
        self.tof.flushOutput()

    def process_data(self, data):
        '''
        Take raw data from serial output and parse it into correct format
        '''
        # Remove ANSI escape charters \x1b[36m (cyan)
        ansi_escape =re.compile(r'(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]')

        # Remove ANSI escape characters
        ansi_removed = ansi_escape.sub('', data)
        # Process into parts
        record = ansi_removed.split(',')
        for value in record:
            key, val = value.split('=')
            if key.strip() == 'Status' and int(val.strip()) == 0:
                continue
            elif key.strip() == 'Distance':
                distance = int(val[:-2].strip())
            else:
                break
        
        return distance

    def get_distance(self, verbose = False):
        '''
        Main operating loop used to read and send data from the spectrometer
        '''
        while True:
            try:
                # Grab the raw data
                raw_data = self.tof.readline()
                # Decode the spectral data
                tof_data = raw_data.decode('utf-8').strip()
                # Process and publish the data
                distance = self.process_data(tof_data)
                break
            except Exception as e:
                if verbose:
                    print('Error in main ToF loop: '+ str(e) + ', trying to readline again')
            
            time.sleep(0.1)
        return distance

    def shutdown(self):
        '''
        Custom shutdown behavior
        '''
        # Close the serial transmission
        self.tof.close()

# # Main functionality
# if __name__ == '__main__':
#     # Initialize the node and name it.
#     try:
#         controller = ToFDriver()
#         dist = controller.get_distance()
#         print(dist)
#     except:
#         controller.shutdown()