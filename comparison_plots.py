#%%
import pandas as pd
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt

# Read in calibration data
hamamatsu_dark = np.median(pd.read_csv('./calibration/hamamatsu_black_ref.csv').to_numpy().astype(np.int32), axis=0)
hamamatsu_white = np.median(pd.read_csv('./calibration/hamamatsu_white_ref.csv').to_numpy().astype(np.int32), axis=0)
mantispectra_dark = np.median(pd.read_csv('./calibration/mantispectra_black_ref.csv').to_numpy()[:,:-5].astype(np.int32), axis=0)
mantispectra_white = np.median(pd.read_csv('./calibration/mantispectra_white_ref.csv').to_numpy()[:,:-5].astype(np.int32), axis=0)

# Create composite calibration file
white_ref = np.concatenate((hamamatsu_white, mantispectra_white))
dark_ref = np.concatenate((hamamatsu_dark, mantispectra_dark))

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

#%%
sample_code_gt = 'A14'       # container-sample code tested on the robot

# sample_code_pred = None    # use if the model's predictions were correct
sample_code_pred = 'A18'     # incorrect contrainer-sample code predicted by the model

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

#%%
train_data_gt = np.apply_along_axis(spectral_calibration, 1, train_data_gt)
demo_data = spectral_calibration(demo_data)
train_data_pred = np.apply_along_axis(spectral_calibration, 1, train_data_pred)

# print(train_data_gt[0, -16:])
#%%

fig = plt.figure()
plt.subplot(1,2,1, ) # spectrapod data
# plot data from dataset (correct code)
for i in range(train_data_gt[:, -16:].shape[0]):
    plt.plot(train_data_gt[i, -16:], c='green')
# plot data from dataset (incorrect predicted code)
if sample_code_pred is not None:
    for i in range(train_data_pred[:, -16:].shape[0]):
        plt.plot(train_data_pred[i, -16:], c='red')
# plot data collected on robot
plt.plot(demo_data[0, -16:], linewidth=3)


plt.subplot(1,2,2) # hamamatsu data
# plot data from dataset (correct code)
for i in range(train_data_gt[:, :-16].shape[0]-1):
    plt.plot(train_data_gt[i, :-16], c='green')
# plot data from dataset (incorrect predicted code)
plt.plot(train_data_gt[i, :-16], c='green', label='train, gt')
if sample_code_pred is not None:
    for i in range(train_data_pred[:, :-16].shape[0]-1):
        plt.plot(train_data_pred[i, :-16], c='red')
    plt.plot(train_data_pred[i, :-16], c='red', label='train, pred')
# plot data collected on robot
plt.plot(demo_data[0, :-16], label='demo', linewidth=3)


plt.legend(loc = 'center right', bbox_to_anchor=(0.7, -.1), ncol=3)
# plt.show()
plt.savefig(f'/home/slurp/Pictures/{sample_code_gt}-{sample_code_pred}.png')

# %%
