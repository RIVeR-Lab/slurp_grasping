# Main imports
import os
import csv
import typing
import os
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
from models import MLP, Simple1DCNN, FusedNet
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold, LeaveOneOut

def build_network(architecture: str, input_size: int, classes: int, hidden_layers: Tuple=(256,128)) -> nn.Module:
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
    D = input_size
    print(f'D: {D}')
    K = classes
    print(f'K: {K}')

    # find device to train on
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device')

    # use model that did well in cross val
    if architecture == 'mlp':
        num_perceptrons = hidden_layers
        num_hidden_layers = len(num_perceptrons)
        use_dropout = True
        model = MLP(D, K, num_hidden_layers, num_perceptrons, 
                    use_dropout).to(device).double()
    elif architecture == '1DCNN':
        use_dropout = True
        model = Simple1DCNN(D, K, use_dropout).to(device).double()
    else:
        raise NotImplementedError("Unsupported network architecture")
        
    return model, device


# Create a dummy array here
def eval_single(model: nn.Module, data: torch.tensor, device: str, le: preprocessing.LabelEncoder) -> None:
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
    output = model(inputs.double())
    print(output)
    # calculate accuracy
    pred_label = torch.argmax(output, dim=1, keepdim=False).cpu()
    print(pred_label)
    print(f'Predicted Class: {le.inverse_transform(pred_label)}')

def main():
    # Acquire a single sample <<TODO>>
    data = torch.tensor(np.random.rand(1,304))
    # Create model
    num_classes = 18
    model, device = build_network('mlp', data.shape[1],num_classes)
    # Load model weights
    model.load_state_dict(torch.load('./weights/all_contents__mlp_best.wts'))
    model.to(device)
    # Load label encoder
    le = preprocessing.LabelEncoder()
    le.classes_ = np.load(f'./weights/label_encoder_class_all_contents.npy')
    eval_single(model,data,device,le)


if __name__ == '__main__':
    main()