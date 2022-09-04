import serial
import re
import time
import numpy as np
import matplotlib.pyplot as plt
import argparse

class SpectrometerDriver():
    def __init__(self, hamamatsu_port_path, spectrapod_port_path):
        self.hama_port_path = hamamatsu_port_path
        self.spectra_port_path = spectrapod_port_path

        self.hama = serial.Serial(self.hama_port_path, baudrate=115200, timeout=0.1)
        self.spectra = serial.Serial(self.spectra_port_path, baudrate=115200, timeout=0.1)
        self.spectra.write(b'fo')  # set spectrapod integration time to the max

        # for removal of ANSI escape characters \x1b[36m (cyan) in spectrapod output
        self.ansi_escape = re.compile(r'(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]')

        self.flush()

    def flush(self):
        self.hama.flushInput()
        self.hama.flushOutput()

        self.spectra.flushInput()
        self.spectra.flushOutput()

    def shutdown(self):
        self.hama.close()
        self.spectra.close()
    
    # borrowed from wesley's code
    def get_data(self):
        print("Collecting spectral data...")

        # (16 channels + 4 parameters) * 5 readings = 100 data points per object
        # param_1: OSR, equivalent to integration time: a value ranging from 1 (high speed) to 15 (low speed)
        # param_2: Internal Lamp power setting: 0 = not activated, 255 = full lamp power
        # param_3: Humidity sensor
        # param_4: Temperature sensor
        hama_total = []

        # collect data twice and throw away the first measurement 
        # do this because sometimes the first set of measurements from hamamatsu is incoherent
        # there is probably a better fix to prevent the problem in the first place, this is just 
        for i in range(2):

            self.flush()
            # spectra_data = self.spectra.readline()

            time.sleep(4.5)  # This might seem long but the arrays wont come fully otherwise
            # spectra_data = self.spectra.readline()
            p = 0

            raw_spectra_data = self.spectra.readline()
            # Decode the raw data
            decoded_data = raw_spectra_data.decode('utf-8')
            # Remove ANSI escape characters
            ansi_removed = self.ansi_escape.sub('', decoded_data)
            # print(ansi_removed)
            # extract first full line of data (20 samples), then capture just the spectral info (16 samples) and leave other data (temp, etc) off
            spectra_data = ansi_removed.split('\r')[1].split()[:-4]

            # read from the hamamatsu until we receive an array that is the correct size
            while len(hama_total) != i+1:
                hama_data = self.hama.readline()
                hama_decoded_data = hama_data.decode()
                hama_listed_data = hama_decoded_data.split(',')

                if len(hama_listed_data) == 289:
                    del hama_listed_data[288]   # remove \n esacpe character from end of hamamatsu output
                    hama_total.append(hama_listed_data)

        self.shutdown()

        spectra_data = np.array(spectra_data).astype(int)
        hama_data = np.array(hama_listed_data).astype(int)
        self.plot_data(hama_data, spectra_data)
        
        # print(hama_data)
        # print(spectra_data)
        # print(len(hama_data))
        # print(len(spectra_data))
        # print(len(hama_data)+len(spectra_data))
        
        all_spectral_data = np.concatenate((hama_data, spectra_data)).reshape((1, -1))
        print(all_spectral_data)
        print('Data Shape:', all_spectral_data.shape)

        return all_spectral_data
    
    def plot_data(self, hama_data, spectra_data):
        plt.figure(figsize=(60, 60))

        plt.subplot(1, 2, 1)
        plt.plot(hama_data)

        plt.subplot(1, 2, 2)
        plt.plot(spectra_data)
        plt.draw()
        plt.show()

    
    def save_spectral_data(self, data, filepath):
        with open(filepath, 'wb') as f:
            np.save(f, data)


# Main functionality
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Capture input from ToF sensor while closing robot gripper')
    parser.add_argument('--hama-port-path', type=str, required=True,
                        help='port path for Hamamatsu spectrometer')
    parser.add_argument('--spectra-port-path', type=str, required=True,
                        help='port path for Spectrapod spectrometer')
    parser.add_argument('--save-filename', type=str, default='../demo_test_data/spectral_data.npy',
                        help='port path for Spectrapod spectrometer')
    args = parser.parse_args()

    spect_controller = SpectrometerDriver(hamamatsu_port_path=args.hama_port_path, spectrapod_port_path=args.spectra_port_path)
    all_spectral_data = spect_controller.get_data()
    spect_controller.save_spectral_data(all_spectral_data, args.save_filename)
