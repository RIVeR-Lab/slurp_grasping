#%%
import pandas as pd
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt

#%%
sample_code_gt = 'G16'       # container-sample code tested on the robot

# sample_code_pred = None    # use if the model's predictions were correct
sample_code_pred = 'G17'     # incorrect contrainer-sample code predicted by the model

# load in data from dataset for the correct comtainer-sample code
training_filename_gt = osp.join('./data', f'{sample_code_gt}.csv')
train_data_gt = pd.read_csv(training_filename_gt)
train_data_gt = train_data_gt.to_numpy()[:, :-5].astype(int).reshape((10,-1))

# load in data collected on robot for the correct comtainer-sample code
demo_filename = osp.join('./demo_data', f'{sample_code_gt}_test.npy')
demo_data = np.load(demo_filename)

# if the model's prediction was incorrect, load in data for the incorrecly predicted container-sample code from the dataset
if sample_code_pred is not None:
    training_filename_pred = osp.join('./data', f'{sample_code_pred}.csv')
    train_data_pred = pd.read_csv(training_filename_pred)
    train_data_pred = train_data_pred.to_numpy()[:, :-5].astype(int).reshape((10,-1))

# print(train_data_gt[0, -16:])

plt.figure()
plt.subplot(1,2,1, ) # spectrapod data
# plot data collected on robot
plt.plot(demo_data[0, -16:], linewidth=3)
# plot data from dataset (correct code)
for i in range(train_data_gt[:, -16:].shape[0]):
    plt.plot(train_data_gt[i, -16:], c='green')
# plot data from dataset (incorrect predicted code)
if sample_code_pred is not None:
    for i in range(train_data_pred[:, -16:].shape[0]):
        plt.plot(train_data_pred[i, -16:], c='red')


plt.subplot(1,2,2) # hamamatsu data
# plot data collected on robot
plt.plot(demo_data[0, :-16], label='demo', linewidth=3)
# plot data from dataset (correct code)
for i in range(train_data_gt[:, :-16].shape[0]-1):
    plt.plot(train_data_gt[i, :-16], c='green')
# plot data from dataset (incorrect predicted code)
plt.plot(train_data_gt[i, :-16], c='green', label='train, gt')
if sample_code_pred is not None:
    for i in range(train_data_pred[:, :-16].shape[0]-1):
        plt.plot(train_data_pred[i, :-16], c='red')
    plt.plot(train_data_pred[i, :-16], c='red', label='train, pred')


plt.legend(loc = 'center right', bbox_to_anchor=(0.7, -.15), ncol=3)
plt.show()

# %%
