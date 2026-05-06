from typing import Dict
import numpy as np
from functools import partial


def flip_bits(bitstring: str, i: int, j: int) -> str:
    """Flips the bits of the bitstrings at two indices.

    Args:
        bitstring (str): Bitstring. len(bitstring) = num_qubits.
        i (int): Index of the first bit.
        j (int): Index of the second bit.

    Returns:
        str: bitstring flipped at the two specified indices.
    """
    b = list(bitstring)
    b[i] = "1" if bitstring[i] == "0" else "0"
    b[j] = "1" if bitstring[j] == "0" else "0"
    return "".join(b)


def state_dict2array(state_dict: Dict[str, complex], size_support: int) -> np.ndarray:
    """Converts the state dictionary of bitstrings and amplitudes to a numpy array.

    Args:
        state_dict (Dict[str, complex]): A quantum state represented by a dictionary mapping bitstrings to amplitudes (Z-basis).
        size_support (int): Size of the support to set to keep the bitstrings. If size_support > |𝛘|, add additional paddings of [-1, -2] to signify NULL.
    Returns:
        np.ndarray: The same quantum state represented by a numpy array of shape (size_support, 2).
    """
    ground_state_list = []

    # Sort the bistring, amplitude pairs by the absolute value of the amplitude squared in a reverse order.
    for i, (k, v) in enumerate(
        sorted(state_dict.items(), key=lambda item: -(abs(item[1]) ** 2))
    ):
        if i == size_support:
            break
        ground_state_list.append([int(k, 2), v])

    # Fill up the padding with [-1., 0.].
    while i + 1 < size_support:
        ground_state_list.append([-1.0, 0.0])
        i += 1

    return np.array(ground_state_list)


def array2state_dict(state_array: np.ndarray, num_spins: int) -> Dict[str, complex]:
    """Converts the state array of size (size_support, 2) to state dictionary.

    Args:
        state_array (np.ndarray): A quantum state represented by a numpy array of shape (size_support, 2).
        num_spins (int): The number of spins.
    Returns:
        Dict[str, complex]: The same quantum state represented by a dictionary mapping bitstrings to amplitudes (Z-basis).
    """

    # Discard all paddings
    state_array_original = state_array[np.abs(state_array[:, 1]) ** 2 > 0.0]
    binary_repr_vec = np.vectorize(partial(np.binary_repr, width=num_spins))
    keys = binary_repr_vec(state_array_original[:, 0].astype(np.int64)).tolist()
    values = state_array_original[:, 1].tolist()

    state_dict = {k: v for k, v in zip(keys, values)}

    return state_dict
