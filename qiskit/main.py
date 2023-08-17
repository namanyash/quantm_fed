# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 15:52:27 2023

@author: Naman Yash
"""

import numpy as np
from qiskit import QuantumCircuit, transpile, Aer
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm
import mpmath
from qiskit.algorithms.optimizers import ADAM, GradientDescent, SPSA
from qiskit.quantum_info import SparsePauliOp, Pauli
from qiskit_aer.primitives import Estimator


n_world = 10
itr = 0
n=8
dataset = 'mnist'
readout_mode = 'softmax'
encoding_mode = 'vanilla'
n_node = 8
k = 48

# backend = provider.get_backend('ibmq_qasm_simulator')
# backend = Aer.get_backend('qasm_simulator')
backend = Aer.get_backend('statevector_simulator')

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
    params_list.append(params)
    

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


def norm(x: np.ndarray) -> np.ndarray:
    mpmath.mp.dps = 50
    
    # Convert numpy array to mpmath matrix
    mp_matrix = mpmath.matrix(x.tolist())
    
    # Normalize the vector using mpmath arithmetic
    norm_factor = mpmath.norm(mp_matrix, 2)  # L2 norm
    normalized_mp_matrix = mp_matrix / norm_factor
    
    # Convert the mpmath matrix back to numpy array
    normalized_x = np.array([float(entry) for row in normalized_mp_matrix.tolist() for entry in row]).ravel()
    
    return normalized_x

## Functions
def clf(params, x, k):
    c = QuantumCircuit(n)
    
    c.initialize(norm(x).tolist())    

    # Circuit gates
    for j in range(k):
        for i in range(n - 1):
            c.cx(i, i + 1)
        for i in range(n):
            c.rx(params[3 * j, i], i)
            c.rz(params[3 * j + 1, i], i)
            c.rx(params[3 * j + 2, i], i)
    print(c)
    return c


def readout(qc, readout_mode='softmax', shots=4000):
    compiled_circuit = transpile(qc, backend)
    
    
    z_operators = [SparsePauliOp(Pauli('Z'*i + 'I' + 'Z'*(n-1-i))) for i in range(n)]

    if readout_mode == 'softmax':
        estimator = Estimator(run_options={"approximation": True, "shots": shots})
        
        # Repeat the compiled circuit for each observable
        circuits = [compiled_circuit] * n
        
        expectation_values = estimator.run(circuits, z_operators).result().values
        # Convert expectation values to real
        logits = np.real(expectation_values)
        
        # Scale logits by 10
        logits *= 10
        
        # Compute softmax
        e_x = np.exp(logits - np.max(logits))
        probs = e_x / e_x.sum()
        
    elif readout_mode == 'sample':
        # Run the circuit to get the statevector
        result = backend.run(compiled_circuit).result()
        statevector = result.get_statevector()
        
        # Compute probabilities from the wavefunction
        wf = np.abs(statevector[:n])**2
        probs = wf / np.sum(wf)
        
    return probs
    

"""
def readout(qc):
    compiled_circuit = transpile(qc, backend)
    result = backend.run(compiled_circuit).result()  ## shots 1024
    qc.measure_all()

    # Get the counts from the measurement
    counts = result.get_counts(qc)
    print(counts)

    # Calculate the expectation values for Z on each qubit
    
    probs = np.array([counts[str(i)] for i in range(n_node)]) / sum(counts.values())
    

    
    
    return probs
"""
"""
def readout(qc, shots=1024):
  backend = Aer.get_backend('aer_simulator')

  if readout_mode == 'softmax':
    logits = []
    for i in range(n_node):
      obs = Z(i)
      expectation = np.real(expectation_value(qc, obs, backend=backend, shots=shots))
      logits.append(expectation * 10)
    probs = np.exp(logits) / np.sum(np.exp(logits))

  elif readout_mode == 'sample':
    job = execute(qc, backend=backend, shots=shots)
    counts = job.result().get_counts()
    states = [int(k, 2) for k in counts.keys()]
    unique, counts = np.unique(states, return_counts=True)
        
    wf = np.zeros(2**n)
    for state, count in zip(unique, counts):
      wf[state] = count
    wf = wf[:n_node]
    probs = wf / np.sum(wf)
        
  return probs
"""
 



def loss(params, x, y, k):
    c = clf(params, x, k)
    probs = readout(c)
    loss   = -np.mean(np.sum(y * np.log(probs + 1e-7), axis=-1))
    return loss
def run_loss_circuit(params, x, y, k):
    global itr
    params = params.reshape(144,8)
    total_loss = 0
    for index in range(x.shape[0]):
        total_loss += loss(params, x[index], y[index], k)
    mean_loss = total_loss/x.shape[0]
    itr += 1
    print(itr, end=' | ')
    return mean_loss

def accuracy(params, x, y, k):
    c = clf(params, x, k)
    probs = readout(c)
    accuracy   = np.argmax(probs, axis=-1) == np.argmax(y, axis=-1)
    return accuracy


def run_accuracy_circuit(params, x, y, k):
    global itr
    params = params.reshape(144,8)
    total_accuracy = 0
    for index in range(x.shape[0]):
        total_accuracy += accuracy(params, x[index], y[index], k)
    mean_accuracy = total_accuracy/x.shape[0]
    return mean_accuracy

loss_list = []
acc_list = []
maxiter=10
#optimizer = ADAM(maxiter=maxiter, lr=1e-2)
# Iterations: (((len(params_list[node])+1)*maxiter)+2)
optimizer = SPSA(maxiter=maxiter)
# Iterations print(51+(maxiter*2))
#optimizer = GradientDescent(maxiter=maxiter)

print(f"Maximum loss evaluations ADAM {(((len(params_list[0].reshape(-1))+1)*maxiter)+2)}")
print(f"Maximum loss evaluations SPSA {51+(maxiter*2)}")
print(f"Optimizer: {optimizer}\n")

for e in tqdm(range(5), leave=False):
    for b in range(100):
        for node in range(n_node-1):
            try:
                x, y = next(iter_list[node])
            except StopIteration:
                iter_list[node] = iter(data_list[node])
                x, y = next(iter_list[node])
            x = x.numpy()
            y = y.numpy()
            #Shape of X array: (128, 256)
            #Shape of Y array: (128, 8)

            ### OPTIMIZER ####
            global itr
            itr = 0
            print(f"\n\nEpoch {e}, batch {b}/{100}, Node {node}")
            print("Optimization start...\n")
            fixed_partial_objective = lambda params: run_loss_circuit(params, x, y, k)
            res   = optimizer.minimize(fixed_partial_objective, params_list[node].reshape(-1))
            print("Optimization complete!\n")
            params_list[node] = res.x
            
            ### Print findings
            print(f'Loss: {res.fun}')
            print(f'Params: {res.x}')
            print(("\n"))

        avg_params = np.mean(np.stack(params_list, axis=0), axis=0)
        for node in range(n_node-1):
            params_list[node] = avg_params
        
        if b % 25 == 0:
            avg_loss = np.mean(loss(run_loss_circuit, x_test[:1024], y_test[:1024], k)[0])
            loss_list.append(avg_loss)
            acc_list.append(run_accuracy_circuit(avg_params, x_test[:1024], y_test[:1024], k).mean())
            print(f"Primary Ouput: Epoch {e}, batch {b}/{100}, loss {avg_loss}, accuracy {acc_list[-1]}")


plt.plot(loss_list)
plt.plot(acc_list)
plt.legend(['loss', 'accuracy'])
plt.ylim(0, 1)
plt.show()