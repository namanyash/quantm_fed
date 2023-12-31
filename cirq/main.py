import numpy as np
import tensorflow_quantum as tfq
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm
import math
import mpmath
import numpy as np
import cirq

n_world = 10

n=8
dataset = 'mnist'
readout_mode = 'softmax'
encoding_mode = 'vanilla'
n_node = 8
k = 48
simulator = cirq.Simulator()

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
    qubits = [cirq.GridQubit(i, 0) for i in range(n)]
    circuit = cirq.Circuit()


    for j in range(k):
        for i in range(n - 1):
            circuit.append(cirq.CNOT(qubits[i], qubits[i + 1]))
        for i in range(n):
            circuit.append(cirq.rx(params[3 * j, i])(qubits[i]))
            circuit.append(cirq.rz(params[3 * j + 1, i])(qubits[i]))
            circuit.append(cirq.rx(params[3 * j + 2, i])(qubits[i]))


    return circuit
"""
def readout(qc, x, readout_mode='softmax', shots=4000):
    state_vector = cirq.final_state_vector(qc, initial_state=norm(x))
    logits = []
    for i in qc.all_qubits():
        z_observable = cirq.Z(i)
        expectation_z = z_observable.expectation_from_state_vector(state_vector, qubit_map={i: pos for pos, i in enumerate(qc.all_qubits())})
        logits.append(expectation_z)

    logits = np.array(logits) * 10  # scaling by 10 as in your example
    probs = np.exp(logits) / sum(np.exp(logits))
    return probs



def loss(params, x, y, k):
    c = clf(params, x, k)
    probs = readout(c, x)
    loss   = -np.mean(np.sum(y * np.log(probs + 1e-7), axis=-1))
    return loss
"""
def readout(qc, x, readout_mode='softmax', shots=4000):
    # Use TFQ for getting the expectation values
    qubits = list(qc.all_qubits())
    z_observables = [cirq.Z(q) for q in qubits]
    expectation_layer = tfq.layers.Expectation()
    expectations = expectation_layer([qc], operators=z_observables, initial_state=norm(x))

    logits = tf.multiply(expectations, 10)
    probs = tf.math.exp(logits) / tf.reduce_sum(tf.math.exp(logits))
    return probs

def loss(params, x, y, k):
    c = clf(params, x, k)
    probs = readout(c, x)
    loss_value = -tf.reduce_mean(tf.reduce_sum(y * tf.math.log(probs + 1e-7), axis=-1))
    return loss_value

min_loss_so_far = 9999
iteration = 0
def run_circuit(params, x, y, k):
    global iteration
    global min_loss_so_far
    total_loss = 0
    new_params = params_tensor.numpy()
    for index in range(x.shape[0]):
        total_loss += loss(new_params, x[index], y[index], k)
    mean_loss = total_loss/x.shape[0]
    iteration += 1
    print(iteration)
    print("mean_loss")
    print(abs(mean_loss))
    if(min_loss_so_far > mean_loss):
      min_loss_so_far = mean_loss
    if(iteration % 10 == 0 ):
      print(min_loss_so_far)
    return tf.convert_to_tensor(abs(mean_loss))

loss_list = []
acc_list = []
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
            print("\n\nShape of X array:", x.shape)
            print("Shape of Y array:", y.shape)

            #a = x.reshape(-1)
            #b = a.reshape(128,256)
            params_tensor = tf.Variable(params_list[node], dtype=tf.float32)
            optimizer = tf.optimizers.Adam(learning_rate=1e-2)

            with tf.GradientTape() as tape:
                current_loss = run_circuit(params_tensor, x, y, k)
            gradients = tape.gradient(current_loss, [params_tensor])
            optimizer.apply_gradients(zip(gradients, [params_tensor]))

            # Extract the optimized parameters after training
            params_list[node] = params_tensor.numpy()

            print(f"Epoch {e}, batch {b}/{100}, Node {node}")
            print(("\n"))