
import json

import numpy as np
import numba
from mlexpt.experiment import run_experiment


def generate_data(mnist_file):
    for line in mnist_file:
        data = json.loads(line)
        pixels = np.array(data['pixels'])
        digit = data['digit']
        yield pixels, digit


@numba.jit
def convert_pixels_to_tnvector(pixels):
    tnvector = np.array([pixels/256., np.sqrt(1-(pixels/256.)*(pixels/256.))]).T
    return tnvector

if __name__ == '__main__':
    pass
