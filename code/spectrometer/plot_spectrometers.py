import table_fields
import matplotlib.pyplot as plt
import pandas as pd

def plot_hamma(dataframe, y, filename):
    values = ["0","1","2","3","4","5","6","7","8","9"]
    color_cycle = ["g", "b", "m", "c", "k", "orange", "pink", "springgreen", "yellow", "brown"]
    plt.figure(figsize=(60, 60))

    for i in range(len(values)):
        print(values)
        plt.plot(dataframe.iloc[i, :].values.tolist(), color=color_cycle[i])

    plt.title(filename, fontdict = {'fontsize': 20})
    plt.xlabel('Channel', fontdict = {'fontsize': 16})
    # plt.xticks(fontsize = 16)
    # plt.yticks(fontsize = 16)
    # plt.ylabel('Photocurrent', fontdict={'fontsize': 16})
    plt.show()
    

def plot_spectrapod(dataframe, y, filename):
    values = ["0","1","2","3","4","5","6","7","8","9"]
    color_cycle = ["g", "b", "m", "c", "k", "orange", "pink", "springgreen", "yellow", "brown"]
    plt.figure(figsize=(60, 60))

    for i in range(len(values)):
        values[i] = dataframe.iloc[i, 0:16].values.tolist()
        plt.plot(values[i], color=color_cycle[i])
        

    plt.title(filename, fontdict = {'fontsize': 10})
    plt.xlabel('Channel', fontdict = {'fontsize': 2})
    plt.xticks(fontsize = 2)
    plt.yticks(fontsize = 2)
    plt.ylabel('Photocurrent', fontdict={'fontsize': 2})
    plt.show()
