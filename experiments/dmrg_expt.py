
import json

import numpy as np
import numba
from mlexpt.experiment import add_multiple_features, run_experiment
from tensorml.dmrg_mnist import QuantumTensorNetworkClassifier


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


def convert_pixels(datum):
    # datum['pixels'] = [list(l) for l in convert_pixels_to_tnvector(np.array([datum['pixels']]))]
    for i, pixel in enumerate(convert_pixels_to_tnvector(np.array([datum['pixels']]))):
        datum['pixel{}'.format(i)] = list(pixel)
    return datum


if __name__ == '__main__':
    feature_adder = add_multiple_features([convert_pixels])
    config = json.load(open('dmrg_mnist_config.json', 'r'))
    config['model']['quantitative_features'] = ['pixel{}'.format(i) for i in range(784)]

    run_experiment(config,
                   feature_adder=feature_adder,
                   model_class=QuantumTensorNetworkClassifier)
