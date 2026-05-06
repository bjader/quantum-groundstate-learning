from typing import Dict

import numpy as np


def szs_from_state_dict(state_dict: Dict[str, complex]):
    n = len(next(iter(state_dict)))

    szs = np.zeros(n)

    for bitstring, amplitude in state_dict.items():
        prob = np.abs(amplitude)**2
        for i, bit in enumerate(bitstring):
            sign = 1 if bit == "0" else -1
            szs[i] += prob * sign * 1/2

    return szs