# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 15:52:27 2023

@author: Naman Yash
"""

import numpy as np
from qiskit import QuantumCircuit, transpile, Aer, execute, assemble
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm
import optax
import math

n_world = 10

n=8
dataset = 'mnist'
readout_mode = 'softmax'
encoding_mode = 'vanilla'
n_node = 8
k = 48

# backend = provider.get_backend('ibmq_qasm_simulator')
backend = Aer.get_backend('qasm_simulator')

print("Start")
if dataset == 'mnist':
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
elif dataset == 'fashion':
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
ind = y_test == 9
x_test, y_test = x_test[~ind], y_test[~ind]
ind = y_test == 8
x_test, y_test = x_test[~ind], y_test[~ind]
ind = y_train == 9
x_train, y_train = x_train[~ind], y_train[~ind]
ind = y_train == 8
x_train, y_train = x_train[~ind], y_train[~ind]

##### x_test (8017, 28, 28) uint8
##### x_train (48200, 28, 28) uint8
##### y_test (8017,) uint8
##### y_train (48200,) uint8
##### ind (54051) bool

x_train = x_train / 255.0
if encoding_mode == 'vanilla':
    mean = 0
elif encoding_mode == 'mean':
    mean = np.mean(x_train, axis=0)
elif encoding_mode == 'half':
    mean = 0.5
x_train = x_train - mean
x_train = tf.image.resize(x_train[..., tf.newaxis], (int(2**(n/2)), int(2**(n/2)))).numpy()[..., 0].reshape(-1, 2**n)
x_train = x_train / np.sqrt(np.sum(x_train**2, axis=-1, keepdims=True))

x_test = x_test / 255.0
x_test = x_test - mean
x_test = tf.image.resize(x_test[..., tf.newaxis], (int(2**(n/2)), int(2**(n/2)))).numpy()[..., 0].reshape(-1, 2**n)
x_test = x_test / np.sqrt(np.sum(x_test**2, axis=-1, keepdims=True))
y_test = np.eye(n_node)[y_test]

##### x_test (8017, 256) float32
##### x_train (48200, 256) float32
##### y_train (48200,) uint8
##### ind (54051) bool
##### y_test (8017,8) float64

print("Processing 1 complete")

def filter_pair(x, y, a, b):
    keep = (y == a) | (y == b)
    x, y = x[keep], y[keep]
    y = np.eye(n_node)[y]
    return x, y

params_list = []
opt_state_list = []
data_list = []
iter_list = []
for node in range(n_node-1):
    x_train_node, y_train_node = filter_pair(x_train, y_train, 0, node + 1)
    data = tf.data.Dataset.from_tensor_slices((x_train_node, y_train_node)).batch(128)
    data_list.append(data)
    iter_list.append(iter(data))

    rng = np.random.default_rng(42)
    key1, key2 = rng.integers(0, 2**31, size=2)
    
    # Generate random numbers using NumPy
    subkey, _ = np.random.default_rng(key2).integers(0, 2**31, size=2)
    params = np.random.default_rng(subkey).normal(size=(3 * k, n))
    opt = optax.adam(learning_rate=1e-2)
    opt_state = opt.init(params)
    params_list.append(params)
    opt_state_list.append(opt_state)
    

##### x_test (8017, 256) float32
##### x_train (48200, 256) float32
##### y_test (8017, 8) float64
##### y_train (48200,) uint8
##### ind (54051) bool
##### x_train_node (12188, 256) float32
##### y_train_node (12188, 8) float64
##### data (size 96) (python.data.ops.batch_op._BatchDataset)
##### data_list (size 7)
##### iter_list (size 7) OwnerIterator Iterator of data
##### params (144, 8) Array of float64
##### params_list size 7 
##### opt = GradientTransformationExtraArgs size 2
##### opt_state tuple size 2
##### opt_state_list (size 7)

print ("Processing done")

## Functions
def clf(params, x, k):
    
    c = QuantumCircuit(n)
    
    # Input
    s = 0;
    for val in x:
        s += val*val
    print(s)
    
    x = x / np.sqrt(np.sum(x**2, axis=-1, keepdims=True))

    s = 0;
    for val in x:
        s += val*val
    print(s)
    # todo: Talk about the error QiskitError: 'Sum of amplitudes-squared does not equal one.' Sum was actually: 1.0000001181440479
    # Same sum of squares after  x = x/np.linalg.norm(x)
    
    
    
    s = 0;
    for val in x:
        s += val*val
    print(s)
    
    
    c.initialize(x);
            
    # Circuit gates
    for j in range(k):
        for i in range(n - 1):
            c.cx(i, i + 1)
        for i in range(n):
            c.rx(params[3 * j, i], i)
            c.rz(params[3 * j + 1, i], i)
            c.rx(params[3 * j + 2, i], i)
    return c

def readout(qc):
    qc.measure_all()
    print("readout start")
    compiled_circuit = transpile(qc, backend)
    qobj = assemble(compiled_circuit)
    result = backend.run(qobj).result()  ## shots 1024
    print("result complete")

    # Get the counts from the measurement
    counts = result.get_counts(qc)
    print(counts)
    
    """
    expected_counts = np.zeros((128,8), dtype=int)

    for key, count in counts.items():
        index = int(key, 2)  # Convert the binary key to an integer index
        expected_counts[index] = count

    total_values = sum(expected_counts)
    if total_values != total_shots:
        scaling_factor = total_shots / total_values
        expected_counts = (expected_counts * scaling_factor).astype(int)
    """
    return counts
    
    
def loss(params, x, y, k):
    print("Loss start")
    c = clf(params, x[0], k)
    return readout(c)
    

loss_list = []
acc_list = []
for node in range(n_node-1):
    try:
        x, y = next(iter_list[node])
    except StopIteration:
        iter_list[node] = iter(data_list[node])
        x, y = next(iter_list[node])
    x = x.numpy()
    y = y.numpy()
    print("Shape of X array:", x.shape)
    print("Shape of Y array:", y.shape)
    

    # Shape of X array: (128, 256)
    # Shape of Y array: (128, 8)
    # Sum of squares of x  = 1;
    
    loss_value = loss(params_list[node], x, y, k)
    print(loss_value)
    
    


