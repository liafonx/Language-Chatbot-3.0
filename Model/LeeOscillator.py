'''
    Copyright:      JarvisLee
    Date:           5/19/2021
    File Name:      LeeOscillator.py
    Description:    The Choatic activation functions named Lee-Oscillator Based on Raymond Lee's paper.
'''

# Import the necessary library.
import math
import os
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torch.autograd as autograd


# Create the class for the Lee-Oscillator.
class LeeOscillator():
    '''
        The Lee-Oscillator based activation function.\n
        Params:\n
            - a (list), The parameters list for Lee-Oscillator of Tanh.\n
            - b (list), The parameters list for Lee-Oscillator of Sigmoid.\n
            - K (integer), The K coefficient of the Lee-Oscillator.\n
            - N (integer), The number of iterations of the Lee-Oscillator.\n
    '''

    # Create the constructor.
    def __init__(self, a=[1, 1, 1, 1, -1, -1, -1, -1], b=[0.6, 0.6, -0.5, 0.5, -0.6, -0.6, -0.5, 0.5], K=50, N=100):
        # Get the parameters for the Lee-Oscillator.
        self.a = a
        self.b = b
        self.K = K
        self.N = N
        # Paths relative to this file's directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        tanh_path = os.path.join(base_dir, 'LeeOscillator-Tanh.csv')
        sigmoid_path = os.path.join(base_dir, 'LeeOscillator-Sigmoid.csv')
        # Draw the bifraction diagram of the Lee-Oscillator.
        if (not os.path.exists(tanh_path)) or (not os.path.exists(sigmoid_path)):
            # Compute the Lee-Oscillator.
            self.TanhCompute(a1=a[0], a2=a[1], a3=a[2], a4=a[3], b1=a[4], b2=a[5], b3=a[6], b4=a[7], K=K, N=N)
            self.SigmoidCompute(a1=b[0], a2=b[1], a3=b[2], a4=b[3], b1=b[4], b2=b[5], b3=b[6], b4=b[7], K=K, N=N)
        # Load precomputed tables and prepare 1D mean curves for fast lookup
        tanh_table_2d = torch.tensor(pd.read_csv(tanh_path).values, dtype=torch.float32)
        sigmoid_table_2d = torch.tensor(pd.read_csv(sigmoid_path).values, dtype=torch.float32)
        self.tanh_table = tanh_table_2d.mean(dim=1)      # shape: [1000]
        self.sigmoid_table = sigmoid_table_2d.mean(dim=1)  # shape: [1000]
        # Precompute constants for interpolation
        self.dx = 0.002
        self.x_min = -1.0
        self.x_max = 1.0
        self.num_bins = int(self.tanh_table.shape[0])

    def _interpolate_table(self, x, table_1d):
        x_clamped = torch.clamp(x, self.x_min, self.x_max)
        idx_float = (x_clamped - self.x_min) / self.dx
        idx0 = torch.floor(idx_float).long().clamp(0, self.num_bins - 2)
        idx1 = idx0 + 1
        w = (idx_float - idx0.to(idx_float.dtype)).to(x.dtype)
        table = table_1d.to(x.device)
        y0 = table.index_select(0, idx0.view(-1)).view_as(x)
        y1 = table.index_select(0, idx1.view(-1)).view_as(x)
        return (1.0 - w) * y0 + w * y1

    # Create the function to apply the Lee-Oscillator of tanh activation function.
    def Tanh(self, x):
        return self._interpolate_table(x, self.tanh_table)

    # Create the function to apply the Lee-Oscillator of sigmoid activation function.
    def Sigmoid(self, x):
        return self._interpolate_table(x, self.sigmoid_table)

    # Create the function to compute the Lee-Oscillator of tanh activation function.
    def TanhCompute(self, a1, a2, a3, a4, b1, b2, b3, b4, K, N):
        # Create the array to store and compute the value of the Lee-Oscillator.
        u = torch.zeros([N])
        v = torch.zeros([N])
        z = torch.zeros([N])
        w = 0
        u[0] = 0.2
        z[0] = 0.2
        Lee = np.zeros([1000, N])
        xAix = np.zeros([1000 * N])
        j = 0
        x = 0
        for i in np.arange(-1, 1, 0.002):
            for t in range(0, N - 1):
                u[t + 1] = torch.tanh(a1 * u[t] - a2 * v[t] + a3 * z[t] + a4 * i)
                v[t + 1] = torch.tanh(b3 * z[t] - b1 * u[t] - b2 * v[t] + b4 * i)
                w = torch.tanh(torch.Tensor([i]))
                # z[t + 1] = (v[t + 1] - u[t + 1]) * np.exp(-K * np.power(i, 2)) + w
                z[t + 1] = (v[t + 1] - u[t + 1]) * np.exp(-K * np.power(i, 2)) + w
                # Store the Lee-Oscillator.
                xAix[j] = i
                j = j + 1
                Lee[x, t] = z[t + 1]
            Lee[x, t + 1] = z[t + 1]
            x = x + 1
        # Store the Lee-Oscillator.
        data = pd.DataFrame(Lee)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data.to_csv(os.path.join(base_dir, 'LeeOscillator-Tanh.csv'))
        plt.figure(1)
        fig = np.reshape(Lee, [1000 * N])
        plt.plot(xAix, fig, ',')
        plt.savefig(os.path.join(base_dir, 'LeeOscillator-Tanh.jpg'))

    # Create the function to compute the Lee-Oscillator of sigmoid activation function.
    def SigmoidCompute(self, a1, a2, a3, a4, b1, b2, b3, b4, K, N):
        # Create the array to store and compute the value of the Lee-Oscillator.
        u = torch.zeros([N])
        v = torch.zeros([N])
        z = torch.zeros([N])
        w = 0
        u[0] = 0.2
        z[0] = 0.2
        Lee = np.zeros([1000, N])
        xAix = np.zeros([1000 * N])
        j = 0
        x = 0
        for i in np.arange(-1, 1, 0.002):
            for t in range(0, N - 1):
                u[t + 1] = torch.tanh(a1 * u[t] - a2 * v[t] + a3 * z[t] + a4 * i)
                v[t + 1] = torch.tanh(b3 * z[t] - b1 * u[t] - b2 * v[t] + b4 * i)
                w = torch.tanh(torch.Tensor([i]))
                z[t + 1] = (v[t + 1] - u[t + 1]) * np.exp(-K * np.power(i, 2)) + w
                # Store the Lee-Oscillator.
                xAix[j] = i
                j = j + 1
                Lee[x, t] = z[t + 1] / 2 + 0.5
            Lee[x, t + 1] = z[t + 1] / 2 + 0.5
            x = x + 1
        # Store the Lee-Oscillator.
        data = pd.DataFrame(Lee)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data.to_csv(os.path.join(base_dir, 'LeeOscillator-Sigmoid.csv'))
        plt.figure(1)
        fig = np.reshape(Lee, [1000 * N])
        plt.plot(xAix, fig, ',')
        plt.savefig(os.path.join(base_dir, 'LeeOscillator-Sigmoid.jpg'))


# Create the main function to test the Lee-Oscillator.
if __name__ == "__main__":
    # Get the parameters list of the Lee-Oscillator.
    a = [-0.2, 0.45, 0.6, 1, 0, -0.55, 0.55, 0]
    b = [0.6, 0.6, -0.5, 0.5, -0.6, -0.6, -0.5, 0.5]
    # Create the Lee-Oscillator's model.
    with autograd.detect_anomaly():
        Lee = LeeOscillator()
        # Test the Lee-Oscillator.
        x = torch.randn((32, 1, 9, 4))
        x = torch.reshape(x, (32, 9, 4, 1))
        for i in range(0, 8):
            print("Oringinal: " + str(x[0][i]))
        x = torch.relu(x)
        for i in range(0, 8):
            print("Relu: " + str(x[0][i]))
        for i in range(0, 8):
            print("Tanh: " + str(Lee.Tanh(x[0][i])))
        for i in range(0, 8):
            print("Sigmoid: " + str(Lee.Sigmoid(x[0][i])))
