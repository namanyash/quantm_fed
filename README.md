# Federated Quantum Machine Learning in Qiskit: Design and Implementation

# Investigators:

- Dr Shiva Pokhrel (shiva.pokhrel@deakin.edu.au) is a Marie Curie Fellow and Senior Mobile and Quantum Computing Lecturer. He received the PhD ICT Engineering degree from Swinburne University of Technology, Australia. He was a research fellow at the University of Melbourne and a Telecom network engineer (2007-2014). His research interests include quantum machine learning, federated learning, semantic communication, federated learning, industry 4.0, blockchain modeling, optimization, recommender systems, 6G, cloud computing, dynamics control, Internet of Things and cyber-physical systems, as well as their applications in intelligent manufacturing, autonomous vehicles and cities. He has been on the editorial board of top-notch journals in the field. He serves/served as the Workshop Chair/Publicity Co-Chair for several IEEE/ACM conferences, including IEEE INFOCOM, IEEE GLOBECOM, IEEE ICC, ACM MobiCom, and more. He was a recipient of the prestigious Marie Curie Fellowship in 2017.

- Dr Jonathan Kua is a Lecturer in Internet of Things and Quantum Computing at Deakin University. He received his B.Eng. (First Class Hons.) degree in telecommunications and network engineering in 2014 and Ph.D. degree in telecommunications engineering in 2019, both from Swinburne University of Technology, Melbourne, Australia. He was previously affiliated with the Centre for Advanced Internet Architectures (2013-2017) and the Internet For Things Research Group (2017-2019), within Swinburne University of Technology. He was awarded a full Netflix Ph.D. scholarship to pursue applied research in high-performance Internet Protocol-based content delivery and adaptive streaming. His research was internationally recognized with a second runner-up of the DASH-IF Best Ph.D. Dissertation Award on "Algorithms and Protocols for Adaptive Content Delivery over the Internet" (2019), and commended locally with the Swinburne Outstanding Thesis Award (2019).

- Naman Yash is a research intern at IoT Lab Deakin working on Federated Quantum Learning implementations in Qiskit. He has industry experience as a software engineer (2020-2022) in the cloud team for JP Morgan Chase, where he worked on developing and deploying cloud-native applications. Naman holds all three AWS associate certifications (Solutions Architect, Developer, and SysOps Administrator) and is a cloud computing enthusiast. He is currently pursuing a Master of IT in Information Technology with majors in cloud computing at Deakin University. Previously, he completed his Bachelor of Engineering in Computer Engineering from Thapar Institute of Technology in India in 2020. At Deakin, his research focuses on leveraging cloud platforms like AWS, IBM, GCP and more to implement quantum computing algorithms and frameworks.

# Quantum Federated Learning Implementations

This repository contains implementations of the quantum federated learning algorithm described in [this paper](https://arxiv.org/abs/2209.00768) using Qiskit and Cirq.

The original [code from the paper](https://github.com/haimengzhao/quantum-fed-infer) uses TensorCircuit, but we have reimplemented it using other frameworks to run on different quantum cloud platforms.

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


