import math
import numpy as np
from matplotlib import pyplot as plt

plot_all_data_boll = False
plot_X_train_boll = False
plot_Y_train_boll = False
plot_Y_train_vs_X_train_boll = True

def plot_all_data(X_train, Y_train):
    plt.plot(np.arange(0, len(X_train)), X_train)
    plt.plot(np.arange(0, len(Y_train)), Y_train)
    plt.xlabel('sensor_reading_n')
    plt.ylabel('sensor_reading_value')
    plt.show()

def plot_X_train(X_train):
    plt.plot(np.arange(0, len(X_train)), X_train)
    plt.xlabel('sensor_reading_n')
    plt.ylabel('sensor_reading_value')
    plt.show()

def plot_Y_train(Y_train):
    plt.plot(np.arange(0, len(Y_train)), Y_train)
    plt.xlabel('sensor_reading_n')
    plt.ylabel('sensor_reading_value')
    plt.show()

def plot_Y_train_vs_X_train(X_train, Y_train):
    n_columns = X_train.shape[1]
    
    n_rows = math.ceil(n_columns / 3)
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5 * n_rows))
    
    axes = np.array(axes).flatten()
    
    for i in range(n_columns):
        axes[i].plot(X_train[:, i], Y_train, 'o')
        axes[i].set_xlabel(f'X_train[:, {i}]')
        axes[i].set_ylabel('Y_train')
        axes[i].set_title(f'Sensor {i}')
        axes[i].grid(True)
    
    for j in range(n_columns, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()
    
def plot_data(X_train, Y_train):
    if plot_all_data_boll == True:
        plot_all_data(X_train, Y_train)
    if plot_X_train_boll == True:
        plot_X_train(X_train)
    if  plot_Y_train_boll == True:
        plot_Y_train(Y_train)
    if plot_Y_train_vs_X_train_boll == True:
        plot_Y_train_vs_X_train(X_train, Y_train)
    
X_train = np.load("X_train.npy")
Y_train = np.load("Y_train.npy")

plot_data(X_train, Y_train)

