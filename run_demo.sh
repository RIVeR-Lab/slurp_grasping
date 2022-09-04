#!/bin/bash

TOF_PORT_PATH='/dev/ttyACM5'
TOF_INSET_DIST='18'
HAMA_PORT_PATH='/dev/ttyACM7'
SPECTRA_PORT_PATH='/dev/ttyUSB3'
SPECTRAL_SAMPLE_FILENAME=$1
echo $SPECTRAL_SAMPLE_FILENAME

# python ./code/demo/grip_to_measure.py --tof-port-path $TOF_PORT_PATH --tof-inset $TOF_INSET_DIST
python ./code/demo/capture_spectral_sample.py --hama-port-path $HAMA_PORT_PATH --spectra-port-path $SPECTRA_PORT_PATH --save-filename $SPECTRAL_SAMPLE_FILENAME
source ../env/bin/activate
python3 ./eval_single.py --spectral-data-path $SPECTRAL_SAMPLE_FILENAME