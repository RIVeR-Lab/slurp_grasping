import serial
import re
import time
import numpy as np
import matplotlib.pyplot as plt
import argparse
from spect import SpectrometerDriver

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

    raw_input("Press Enter to collect spectral sample...")

    spect_controller = SpectrometerDriver(hamamatsu_port_path=args.hama_port_path, spectrapod_port_path=args.spectra_port_path)
    all_spectral_data = spect_controller.get_data()
    spect_controller.save_spectral_data(all_spectral_data, args.save_filename)
