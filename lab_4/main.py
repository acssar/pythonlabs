import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-file1', dest='f1', help="first matrix file", default="file_1.txt")
parser.add_argument('-file2', dest='f2', help="second matrix file", default="file_2.txt")
parser.add_argument('-prob', dest='p', help="probability of synthetic data", default=0.2)
args = parser.parse_args()


def numpy_where(orig, synth, p):
    """

    Parameters
    ----------
    orig: numpy.ndarray
        Original data array.
    synth: numpy.ndarray
        Synthetic data array.
    p: float
        Probability of synthetic data choice.

    Returns
    -------
    numpy.ndarray with selection result

    """
    return np.where(np.random.random(len(orig)) > p, orig, synth)


def numpy_choose(orig, synth, p):
    """

    Parameters
    ----------
    orig: numpy.ndarray
        Original data array.
    synth: numpy.ndarray
        Synthetic data array.
    p: float
        Probability of synthetic data choice.

    Returns
    -------
    numpy.ndarray with selection result

    """
    arr = (np.random.random(len(orig)) > p)
    return np.choose(arr, [synth, orig])


original_data = np.loadtxt(args.f1, dtype=int)
synthetic_data = np.loadtxt(args.f2, dtype=int)
print("selection using choose: ", numpy_choose(original_data, synthetic_data, args.p))
print("selection using where: ", numpy_where(original_data, synthetic_data, args.p))
