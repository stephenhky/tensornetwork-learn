
import json

import numpy as np
import numba
import tensorflow as tf
import tensornetwork as tn


class QuantumDMRGLayer(tf.keras.layers.Layer):
    def __init__(self, dimvec, pos_label, nblabels, bond_len, nearzero_std=1e-9):
        super(QuantumDMRGLayer, self).__init__()
        self.dimvec = dimvec
        self.pos_label = pos_label
        self.nblabels = nblabels
        self.m = bond_len

        self.mps_tensors = [tf.Variable(self.mps_tensor_initial_values(i, nearzero_std=nearzero_std),
                                        trainable=True,
                                        name='mps_tensors_{}'.format(i))
                            for i in range(self.dimvec)]

    def mps_tensor_initial_values(self, idx, nearzero_std=1e-9):
        if idx == 0 or idx == self.dimvec - 1:
            tempmat = tf.eye(max(2, self.m))
            mat = tempmat[0:2, :] if 2 < self.m else tempmat[:, 0:self.m]
            return mat + tf.random.normal(mat.shape, mean=0.0, stddev=nearzero_std)
        elif idx == self.pos_label:
            return tf.random.normal((2, self.m, self.m, self.nblabels),
                                    mean=0.0,
                                    stddev=nearzero_std)
        else:
            return tf.random.normal((2, self.m, self.m),
                                    mean=0.0,
                                    stddev=nearzero_std)

    def infer_single(self, input):
        assert input.shape[0] == self.dimvec
        assert input.shape[1] == 2

        nodes = [
            tn.Node(self.mps_tensors[i], backend='tensorflow')
            for i in range(self.dimvec)
        ]
        input_nodes = [
            tn.Node(input[i, :], backend='tensorflow')
            for i in range(self.dimvec)
        ]

        for i in range(self.dimvec):
            nodes[i][0] ^ input_nodes[i][0]
        nodes[0][1] ^ nodes[1][1]
        for i in range(1, self.dimvec - 1):
            nodes[i][2] ^ nodes[i + 1][1]

        final_node = tn.contractors.auto(nodes + input_nodes,
                                         output_edge_order=[nodes[self.pos_label][3]])
        return final_node.tensor

    def call(self, inputs):
        return tf.vectorized_map(self.infer_single, inputs)


def generate_data(mnist_file):
    for line in mnist_file:
        data = json.loads(line)
        pixels = np.array(data['pixels'])
        digit = data['digit']
        yield pixels, digit


@numba.njit(numba.float64[:, :](numba.float64[:]))
def convert_pixels_to_tnvector(pixels):
    tnvector = np.concatenate(
        (np.expand_dims(np.cos(0.5*np.pi*pixels/256.), axis=0),
         np.expand_dims(np.sin(0.5*np.pi*pixels/256.), axis=0)),
        axis=0
    ).T
    return tnvector


def convert_pixels(datum):
    # datum['pixels'] = [list(l) for l in convert_pixels_to_tnvector(np.array([datum['pixels']]))]
    for i, pixel in enumerate(convert_pixels_to_tnvector(np.array([datum['pixels']]))):
        datum['pixel{}'.format(i)] = list(pixel)
    return datum


def QuantumKerasModel(dimvec, pos_label, nblabels, bond_len, nearzero_std=1e-9, optimizer='adam'):
    quantum_dmrg_model = tf.keras.Sequential([
        tf.keras.Input(shape=(dimvec, 2)),
        QuantumDMRGLayer(dimvec=dimvec,
                         pos_label=pos_label,
                         nblabels=nblabels,
                         bond_len=bond_len,
                         nearzero_std=nearzero_std),
        tf.keras.layers.Softmax()
    ])
    quantum_dmrg_model.compile(optimizer=optimizer, loss=tf.keras.losses.CategoricalCrossentropy())
    return quantum_dmrg_model


if __name__ == '__main__':
    # model parameters
    dimvec = 784
    pos_label = 392
    nblabels = 10
    bond_len = 10
    nbdata = 70000

    # training and CV parameters
    nb_epochs = 10
    cv_fold = 5
    batch_size = 10

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

        print('Number of training data: {}'.format(trainX.shape[0]))
        print('Number of test data: {}'.format(testX.shape[0]))

        # Initializing Keras model
        print('Initializing Keras model...')
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        quantum_dmrg_model = QuantumKerasModel(dimvec, pos_label, nblabels, bond_len, optimizer=optimizer)

        print(quantum_dmrg_model.summary())

        # Training
        print('Training')
        quantum_dmrg_model.fit(trainX, trainY, batch_size=batch_size, epochs=nb_epochs)

        # Testing
        print('Testing')
        predictedY = quantum_dmrg_model.predict(testX)
        cross_entropy = - np.sum(testY*np.log(predictedY), axis=1)
        print('Cross-entropy = {}'.format(cross_entropy))
        nbmatches = np.sum(np.argmax(testY, axis=1) == np.argmax(predictedY, axis=1))
        print('Number of matches = {}'.format(nbmatches))
        print('Accuracy = {:.2f}%'.format(nbmatches/nbdata*100))

        accuracies.append(nbmatches/nbdata)

    print('Average accuracy = {:.2f}%'.format(np.mean(accuracies)*100))
