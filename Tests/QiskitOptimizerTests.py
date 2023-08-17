# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 13:35:07 2023

@author: Naman Yash
"""


import numpy as np
from qiskit.algorithms.optimizers import SPSA, ADAM
itr = 0
def objective(x):
    print(x)
    global itr
    itr += 1
    value = (np.linalg.norm(x) + 1*np.random.rand(1))
    print(value)
    return value

class TerminationChecker:

    def __init__(self, N : int):
        self.N = N
        self.values = []

    def __call__(self, nfev, parameters, value, stepsize, accepted) -> bool:
        self.values.append(value)

        if len(self.values) > self.N:
            last_values = self.values[-self.N:]
            pp = np.polyfit(range(self.N), last_values, 1)
            slope = pp[0] / self.N

            if slope > 0:
                return True
        return False

#[0.0760747  0.09873245]
#[0.15605202]
#spsa = ADAM(maxiter=maxiter, lr=0.01)

maxiter = 76
spsa = SPSA(maxiter=maxiter)
x = [0.5,0.5]
res = spsa.minimize(objective, x)
print(51+(maxiter*2))
print(itr)
print(res)
#print(parameters)
#print(value)
#print(f'SPSA completed after {niter} iterations')