# Federated Quantum Machine Learning in Qiskit: Design and Implementation

# Quantum Federated Learning Implementations

This repository contains implementations of the quantum federated learning algorithm: [this paper](https://arxiv.org/abs/2209.00768) using Qiskit and Cirq.

The original [code from the paper](https://github.com/haimengzhao/quantum-fed-infer) uses TensorCircuit.

## Nontrivial Challenges

## Contents

The `qiskit` and `cirq` directories contain work-in-progress implementations of the algorithm using those frameworks.

The `/test` directory contains various test scripts and programs used for learning quantum computing concepts and testing components.

## About

This code allows testing and comparison of the quantum federated learning algorithm on different quantum cloud providers by reimplementing with different frameworks.

The implementations are still in development and aim to eventually reproduce the results from the original paper using Qiskit and Cirq. Other libraries like Q# may be added in the future as the scope expands to more cloud platforms.

## Usage

The main files to run will be `main.py` in the `qiskit` and `cirq` directories once complete. For now they are still under development.

The tests and examples in `/test` can be run individually as needed.

## Requirements

- Qiskit
- Cirq
- Python 3
- Packages to be listed in `requirements.txt` (to be added)

## Diagrams

The following diagram demonstrates the implementation of the qFedAvg algorithm utilizing Qiskit:
![Code explanation diagram](https://github.com/namanyash/quantm_fed/blob/main/diagrams/CodeExp.png)

The following diagram demonstrates the procedure for computing the loss within the optimization process of the designated loss function.
![Loss function explanation diagram](https://github.com/namanyash/quantm_fed/blob/main/diagrams/circuit.png)
