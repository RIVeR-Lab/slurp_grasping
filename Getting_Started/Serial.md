---
layout: default
title: Serial
parent: Getting Started
---

# Serial Without ROS

To gather data from serial without ROS you will need the following libraries:

1. Pyserial
```bash
python -m pip install pyserial
```
or
```bash
conda install pyserial
```
2. Numpy
```bash
pip install numpy
```
or
```bash
conda install numpy
```
3. Pandas
```bash
pip install pandas
```
or
```bash
conda install pandas
```
4. Matplotlib
```bash
pip install matplotlib
```
or
```bash
conda install matplotlib
```


### Using the Spectrometers
{: .note }
You might have to change the port paths to get the spectrometers working 
```bash
cd slurp_grasping/code/data_collect
```
For the OEM Sensing Board
```bash
python spectrapod_new.py
```
For the C12880MA
```bash
python3 hamamatsu_new.py
```
For both
```bash
python3 data_collect.py
```