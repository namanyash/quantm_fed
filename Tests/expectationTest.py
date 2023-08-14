# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 11:20:11 2023

@author: Naman Yash
"""

import numpy as np
from qiskit import *
from qiskit.providers.fake_provider import FakeAthens
import mthree

backend = FakeAthens()

ghz2 = QuantumCircuit(2)
ghz2.h(0)
ghz2.cx(0,1)
ghz2.measure_all()

trans_ghz2 = transpile(ghz2, backend)

ghz3 = QuantumCircuit(3)
ghz3.h(0)
ghz3.cx(0,1)
ghz3.cx(1,2)
ghz3.measure_all()

trans_ghz3 = transpile(ghz3, backend)

raw2 = backend.run(trans_ghz2, shots=4000).result().get_counts()
raw3 = backend.run(trans_ghz3, shots=4000).result().get_counts()

print(raw3)
mit = mthree.M3Mitigation(backend)
mit.cals_from_system()

quasi2 = mit.apply_correction(raw2, [0,1], return_mitigation_overhead=True)
quasi3 = mit.apply_correction(raw3, [0,1,2], return_mitigation_overhead=True)

print('GHZ2:', quasi2.expval())
print('GHZ3:', quasi3.expval())