# Main imports
import os
import csv
import typing
import os
import pickle as pk
import time
import torch
import joblib
from torch import nn
from copy import deepcopy
from sklearn.metrics import ConfusionMatrixDisplay
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
from datetime import datetime
from sklearn import preprocessing
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import matplotlib
from IPython.display import display
import matplotlib.pyplot as plt
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter
from utils import SpectralData, train_epochs
from models import MLP, Simple1DCNN, FusedNet, ContNet
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold, LeaveOneOut
import argparse
from code.demo.spect import SpectrometerDriver


def build_network(input_size: int, num_containers: int, num_contents: int, hidden_layers_l1: Tuple=(256,128), hidden_layers_l2: Tuple=(256,128)) -> Tuple[nn.Module,str]:
    '''
    Generate a train able neural network architecture
    
    Args: 
        architecture (str): Type of architecture to generate
        input_size (int): Number of input features
        classes (int): Number of output features (classes)
        hidden_layers (Tuple, optional): Size of sequential hidden layers to use in network construction
        
    Returns:
        nn.Module: Model architecture
    '''

    # find device to train on
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device')

    model = ContNet(input_size, num_containers, num_contents, hidden_layers_l1, hidden_layers_l2)
    
    return model, device


# Create a dummy array here
def eval_single(model: nn.Module, data: torch.tensor, device: str, le_contents, le_containers) -> None:
    '''
    Calculate test accuracy using held-out dataset
    Args: 
        model (nn.Module): Model architecture to pass data through
        test_loader (DataLoader): PyTorch dataloader with references to test data
        device (str): Location to execute computations over
        le (preproccessing.LabelEncoder): Object used to translate string labels into numeric indicies        
    Returns:
        None
    '''
    # Evaluate model on held out test set
    model.train(False)

    inputs = data.to(device)
    container_out, content_out = model(inputs)

    m = nn.Softmax(dim=1)

    # calculate accuracy
    pred_containers = torch.argmax(container_out, dim=1, keepdim=False).cpu()
    pred_contents = torch.argmax(content_out, dim=1, keepdim=False).cpu()
    print(sorted(zip(le_containers.classes_,m(container_out).cpu().numpy()[0,:]), key=lambda x: x[1], reverse=True)[:3])
    print(f'Predicted Container Class: {le_containers.inverse_transform(pred_containers)}')
    print(sorted(zip(le_contents.classes_,m(content_out).cpu().numpy()[0,:]), key=lambda x: x[1], reverse=True)[:3])
    print(f'Predicted Contents Class: {le_contents.inverse_transform(pred_contents)}')

# Read in calibration data
hamamatsu_dark = np.median(pd.read_csv('./calibration/hamamatsu_black_ref.csv').to_numpy().astype(np.int32), axis=0)
hamamatsu_white = np.median(pd.read_csv('./calibration/hamamatsu_white_ref.csv').to_numpy().astype(np.int32), axis=0)
mantispectra_dark = np.median(pd.read_csv('./calibration/mantispectra_black_ref.csv').to_numpy()[:,:-5].astype(np.int32), axis=0)
mantispectra_white = np.median(pd.read_csv('./calibration/mantispectra_white_ref.csv').to_numpy()[:,:-5].astype(np.int32), axis=0)

# Create composite calibration file
white_ref = np.concatenate((hamamatsu_white, mantispectra_white))[1:]
dark_ref = np.concatenate((hamamatsu_dark, mantispectra_dark))[1:]

# Create calibration function
def spectral_calibration(reading):
    t = np.divide((reading-dark_ref), (white_ref-dark_ref), where=(white_ref-dark_ref)!=0)
    # Handle cases where there is null division, which casts values as "None"
    if np.sum(t==None) > 0:
        print('Null readings!')
    t[t== None] = 0
    # Handle edge cases with large spikes in data, clip to be within a factor of the white reference to avoid skewing the model
    t = np.clip(t,-2,2)
    return t

def main(args, spect_controller):


    # 304 length vector, 288 from hamamatsu, 16 from mantispectra
    num_containers = 6
    num_contents = 13
    model, device = build_network(606, num_containers, num_contents, hidden_layers_l1=(200,100,50), hidden_layers_l2=(200,100,50))
    model.load_state_dict(torch.load('./weights/cont_net_with_ibuprofen.wts', map_location=torch.device(device)))
    model.to(device)

    model.eval()
    # Load label encoder
    # le_containers = preprocessing.LabelEncoder()
    # le_containers.classes_ = np.load(f'./weights/label_encoder_class_all_containers.npy')

    # raw_datum = np.load('/home/slurp/git/slurp_demo/slurp_grasping/demo_real_time_data/K11_rt_test.npy')


# Fit labels to model|
    le_contents = preprocessing.LabelEncoder()
    le_containers = preprocessing.LabelEncoder()
    
    le_contents.classes_ = np.load('./weights/label_encoder_class_fusion_contents.npy')
    le_containers.classes_ = np.load('./weights/label_encoder_class_fusion_containers.npy')
    
    save_samples = []
    count = 0
    while True:
        
        raw_data = spect_controller.get_data()
        # raw_data = raw_datum[count]
        # count += 1

        with torch.no_grad():
            data = spectral_calibration(raw_data[:,1:])
            data = torch.Tensor(np.hstack((data,np.gradient(data,axis=1))))
            print(len(data))
            # Get the container
            eval_single(model, data, device, le_contents, le_containers)
            # From the contianer run the most likely content
            # print(container)
            # # Create model
            # model_contents.load_state_dict(torch.load(f'./weights/all_contents_{container}__mlp_best.wts', map_location=torch.device(device)))
            # model_contents.to(device)
            # model_contents.eval()
            # le_contents = preprocessing.LabelEncoder()
            # le_contents.classes_ = np.load(f'./weights/label_encoder_class_all_contents_{container}.npy')
            # print('=============')
            # contents = eval_single(model_contents,data,device,le_contents)
            time.sleep(1)

            save_samples.append(raw_data)
            if len(save_samples) == 5:
                spect_controller.save_spectral_data(save_samples, args.spectral_data_path)
                print()
                print("FIVE SAMPLES SAVED")
                print()
                input("Press Enter to continue spectral sample collection...")
    
    spect_controller.shutdown()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Save spectral data')
    parser.add_argument('--spectral-data-path', type=str, default='../demo_test_data/spectral_data.npy',
                        help='port path for Spectrapod spectrometer')
    parser.add_argument('--hama-port-path', type=str, required=True,
                        help='port path for Hamamatsu spectrometer')
    parser.add_argument('--spectra-port-path', type=str, required=True,
                        help='port path for Spectrapod spectrometer')
    args = parser.parse_args()

    input("Press Enter to collect spectral sample...")
    spect_controller = SpectrometerDriver(hamamatsu_port_path=args.hama_port_path, spectrapod_port_path=args.spectra_port_path)

    main(args, spect_controller)