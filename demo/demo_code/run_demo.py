import subprocess
import argparse

tof_port_path = '/dev/ttyACM5'
tof_inset_dist = '18'
hamamatsu_port_path = '/dev/ttyACM7'
spectrapod_port_path = '/dev/ttyUSB3'

parser = argparse.ArgumentParser(description='Save spectral data')
parser.add_argument('--save_filename', type=str, default='../demo_test_data/spectral_data.npy',
                    help='port path for Spectrapod spectrometer')
args = parser.parse_args()


subprocess.call(['python', './code/demo/grip_to_measure.py', '--tof-port-path', tof_port_path, '--tof-inset', tof_inset_dist])
subprocess.call(['python', './code/demo/capture_spectral_sample.py', '--hama-port-path', hamamatsu_port_path, '--spectra-port-path', spectrapod_port_path])
subprocess.call(['python3', './eval_single.py', '--spectral-data-path', args.save_filename])