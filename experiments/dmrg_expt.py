
import json

import numpy as np
import numba
import tensorflow as tf
# from mlexpt.experiment import add_multiple_features, run_experiment

# from tensorml.dmrg_mnist import QuantumTensorNetworkClassifier
from tensorml.dmrg_mnist import QuantumDMRGLayer


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


def QuantumKerasModel(dimvec, pos_label, nblabels, bond_len, unihigh=0.05, optimizer='adam'):
    quantum_dmrg_model = tf.keras.Sequential([
        tf.keras.Input(shape=(dimvec, 2)),
        QuantumDMRGLayer(dimvec=dimvec,
                         pos_label=pos_label,
                         nblabels=nblabels,
                         bond_len=bond_len,
                         unihigh=unihigh),
        tf.keras.layers.Softmax()
    ])
    quantum_dmrg_model.compile(optimizer=optimizer, loss=tf.keras.losses.CategoricalCrossentropy())
    return quantum_dmrg_model


if __name__ == '__main__':
    # model parameters
    dimvec = 784
    pos_label = 392
    nblabels = 10
    bond_len = 20
    nbdata = 70000

    # training and CV parameters
    nb_epochs = 10
    cv_fold = 5

    # Prepare for cross-validation
    cv_labels = np.random.choice(range(cv_fold), size=nbdata)

    # Reading the data
    label_dict = {str(i): i for i in range(10)}
    X = np.zeros((nbdata, dimvec, 2))
    Y = np.zeros((nbdata, nblabels))
    for i, (pixels, label) in enumerate(generate_data(open('mnist_784/mnist_784.json', 'r'))):
        X[i, :, :] = convert_pixels_to_tnvector(pixels)
        Y[i, label_dict[label]] = 1.

    # cross_validation
    accuracies = []
    for cv_idx in range(cv_fold):
        print('Round {}'.format(cv_idx))
        trainX = X[cv_labels!=cv_idx, :, :]
        trainY = Y[cv_labels!=cv_idx, :]
        testX = X[cv_labels==cv_idx, :, :]
        testY = Y[cv_labels==cv_idx, :]

        print(trainX.shape)
        print(trainY.shape)
        print(testX.shape)
        print(testY.shape)

        # Initializing Keras model
        print('Initializing Keras model...')
        quantum_dmrg_model = QuantumKerasModel(dimvec, pos_label, nblabels, bond_len)

        # Training
        print('Training')
        quantum_dmrg_model.fit(trainX, trainY, epochs=nb_epochs)

        # Testing
        print('Testing')
        predictedY = quantum_dmrg_model.predict(testX)
        cross_entropy = - np.sum(testY*np.log(predictedY), axis=1)
        print('Cross-entropy = {}'.format(cross_entropy))
        nbmatches = np.sum(np.argmax(testY, axis=1) == np.argmax(predictedY, axis=1))
        print('Number of matches = {}'.format(nbmatches))
        print('Accuracy = {.2f}%'.format(nbmatches/nbdata*100))

        accuracies.append(nbmatches/nbdata)

    print('Average accuracy = {.2f}%'.format(np.mean(accuracies)*100))
