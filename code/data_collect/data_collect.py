# from msilib.schema import tables
from hamamatsu_new import HamamatsuDriver
from spectrapod_new import SpectrapodDriver
from slurp_grasping.code.demo.tof import ToFDriver
from table_fields import ALL_FIELDS
import time
import numpy as np
import pandas as pd
import argparse


hamamatsu_port_path = '/dev/ttyACM6'
spectrapod_port_path = '/dev/ttyUSB3'
tof_port_path = '/dev/ttyACM7'


hama = HamamatsuDriver(port_path=hamamatsu_port_path, int_time_ms=512)
spectra = SpectrapodDriver(port_path=spectrapod_port_path, int_time_ms=512)
tof = ToFDriver(port_path=tof_port_path)

parser = argparse.ArgumentParser(description='Save spectral data')
parser.add_argument('--save_filename', type=str, default='../demo_test_data/spectral_data.npy',
                    help='port path for Spectrapod spectrometer')
args = parser.parse_args()


# print('Input filename:')
filename =  f'{args.save_filename}.csv'


num_samples = 50

t0 = time.time()

samples = []

for i in range(num_samples):

    print()
    print(f"{i}: Change the orientation! - Press ctrl+c to continue...")
    change = input()

    hama.flush()
    spectra.flush()

    hama_sample = hama.capture_sample()
    print(hama_sample)
    spectra_sample = spectra.capture_sample()
    print(spectra_sample)
    tof_dist = tof.get_distance()
    print(tof_dist)

    # sample = np.concatenate((hama_sample, spectra_sample, [tof_dist]))
    sample = np.concatenate((hama_sample, spectra_sample))
    # print(sample)

    samples.append(sample)

df = pd.DataFrame(samples, columns=ALL_FIELDS)

df.to_csv(filename)

print(time.time()- t0)