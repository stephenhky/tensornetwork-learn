
import io
import json
import argparse
import time

import numpy as np
import numba
import tensorflow as tf
import tensornetwork as tn


class QuantumDMRGLayer(tf.keras.layers.Layer):
    def __init__(self, dimvec, pos_label, nblabels, bond_len, nearzero_std=1e-9, isolated_labelnode=True):
        super(QuantumDMRGLayer, self).__init__()
        self.dimvec = dimvec
        self.pos_label = pos_label
        self.nblabels = nblabels
        self.m = bond_len
        self.isolated_label = isolated_labelnode

        assert self.pos_label >= 0 and self.pos_label < self.dimvec

        self.mps_tensors = [tf.Variable(self.mps_tensor_initial_values(i, nearzero_std=nearzero_std),
                                        trainable=True,
                                        name='mps_tensors_{}'.format(i))
                            for i in range(self.dimvec)]
        if self.isolated_label:
            self.output_tensor = tf.Variable(tf.random.normal((self.m, self.m, self.nblabels),
                                                              mean=0.0,
                                                              stddev=nearzero_std),
                                             trainable=True,
                                             name='mps_output_node')

    def mps_tensor_initial_values(self, idx, nearzero_std=1e-9):
        if idx == 0 or idx == self.dimvec - 1:
            tempmat = tf.eye(max(2, self.m))
            mat = tempmat[0:2, :] if 2 < self.m else tempmat[:, 0:self.m]
            return mat + tf.random.normal(mat.shape, mean=0.0, stddev=nearzero_std)
        elif not self.isolated_label and idx == self.pos_label:
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
        if self.isolated_label:
            output_node = tn.Node(self.output_tensor, backend='tensorflow')
        input_nodes = [
            tn.Node(input[i, :], backend='tensorflow')
            for i in range(self.dimvec)
        ]

        for i in range(self.dimvec):
            nodes[i][0] ^ input_nodes[i][0]
        if self.isolated_label:
            nodes[0][1] ^ nodes[1][1]
            for i in range(1, self.pos_label):
                nodes[i][2] ^ nodes[i + 1][1]
            nodes[self.pos_label][2] ^ output_node[0]
            output_node[1] ^ nodes[self.pos_label + 1][1]
            for i in range(self.pos_label + 1, self.dimvec - 1):
                nodes[i][2] ^ nodes[i + 1][1]
        else:
            nodes[0][1] ^ nodes[1][1]
            for i in range(1, self.dimvec-1):
                nodes[i][2] ^ nodes[i + 1][1]

        if self.isolated_label:
            final_node = tn.contractors.auto(nodes + input_nodes + [output_node],
                                             output_edge_order=[output_node[2]])
        else:
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


def QuantumKerasModel(dimvec, pos_label, nblabels, bond_len, nearzero_std=1e-9, optimizer='adam'):
    quantum_dmrg_model = tf.keras.Sequential([
        tf.keras.Input(shape=(dimvec, 2)),
        QuantumDMRGLayer(dimvec=dimvec,
                         pos_label=pos_label,
                         nblabels=nblabels,
                         bond_len=bond_len,
                         nearzero_std=nearzero_std),
        tf.keras.layers.LayerNormalization(beta_initializer='RandomUniform',
                                           gamma_initializer='RandomUniform'),
        tf.keras.layers.Softmax()
    ])
    quantum_dmrg_model.compile(optimizer=optimizer, loss=tf.keras.losses.CategoricalCrossentropy())
    return quantum_dmrg_model


# Not used.
def DenseTNKerasModel(dimvec, hidden_dim, nblabels, bond_len, nearzero_std=1e-9, optimizer='adam'):
    tn_model = tf.keras.Sequential([
        tf.keras.Input(shape=(dimvec, 2)),
        tf.keras.layers.Reshape((dimvec*2,)),
        tf.keras.layers.Dense(hidden_dim*2, activation=None),
        tf.keras.layers.Reshape((hidden_dim, 2)),
        QuantumDMRGLayer(dimvec=hidden_dim,
                         pos_label=hidden_dim // 2,
                         nblabels=nblabels,
                         bond_len=bond_len,
                         nearzero_std=nearzero_std),
        tf.keras.layers.Softmax()
    ])
    tn_model.compile(optimizer=optimizer, loss=tf.keras.losses.CategoricalCrossentropy())
    return tn_model


# From: https://stackoverflow.com/questions/41665799/keras-model-summary-object-to-string
def get_keras_model_summary(model):
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\n'))
    summary_string = stream.getvalue()
    stream.close()
    return summary_string


def get_argparser():
    argparser = argparse.ArgumentParser(description='Testing Quantum Tensor Network Model')
    argparser.add_argument('bond_len', type=int, help='bond length')
    argparser.add_argument('nb_epochs', type=int, help='number of epochs')
    argparser.add_argument('batch_size', type=int, help='batch size')
    argparser.add_argument('learning_rate', type=float, help='learning rate of Adam optimizer')
    argparser.add_argument('--cv_fold', type=int, default=5, help='number of cross-validation folds')
    argparser.add_argument('--pos_label', type=int, default=392, help='position of label node')
    argparser.add_argument('--std', type=float, default=1e-4, help='near zero initialization matrix noise')
    argparser.add_argument('--output_file', default=None, help='output log')
    return argparser


if __name__ == '__main__':
    argparser = get_argparser()
    args = argparser.parse_args()

    outputfilename = args.output_file

    # model parameters
    dimvec = 784
    pos_label = args.pos_label
    nblabels = 10
    bond_len = args.bond_len
    nbdata = 70000

    # training and CV parameters
    nb_epochs = args.nb_epochs
    cv_fold = args.cv_fold
    batch_size = args.batch_size
    std = args.std
    learning_rate = args.learning_rate

    # printing out
    hypparam_strtoprint = 'Number of pixels: {}\n'.format(dimvec) + \
                 'Position of label node: {}\n'.format(pos_label) + \
                 'Number of labels: {}\n'.format(nblabels) + \
                 'Bond length: {}\n'.format(bond_len) + \
                 'Number of epochs: {}\n'.format(nb_epochs) + \
                 'Batch size: {}\n'.format(batch_size) + \
                 'Noise: {}\n'.format(std) + \
                 'Learning rate of Adam optimizer: {}\n'.format(learning_rate) + \
                 'Number of GPUs available: {}\n'.format(len(tf.config.list_physical_devices('GPU')))
    print(hypparam_strtoprint)
    if outputfilename is not None:
        outputfile = open(outputfilename, 'w')
        outputfile.write(hypparam_strtoprint)
        outputfile.close()

    # Time start
    starttime = time.time()

    # Prepare for cross-validation
    cv_labels = np.random.choice(range(cv_fold), size=nbdata)

    # Reading the data
    label_dict = {str(i): i for i in range(10)}
    X = np.zeros((nbdata, dimvec, 2))
    Y = np.zeros((nbdata, nblabels))
    for i, (pixels, label) in enumerate(generate_data(open('mnist_784/mnist_784.json', 'r'))):
        X[i, :, :] = convert_pixels_to_tnvector(pixels)
        Y[i, label_dict[label]] = 1.

    endreaddata_time = time.time()

    # cross_validation
    if outputfilename is not None:
        with open(outputfilename, 'a') as outputfile:
            outputfile.write('Cross-Validation\n')
            outputfile.write('================\n')

    accuracies = []
    cv_time_records = [time.time()]
    for cv_idx in range(cv_fold):
        print('Round {}'.format(cv_idx))
        if outputfilename is not None:
            with open(outputfilename, 'a') as outputfile:
                outputfile.write('Round {}\n'.format(cv_idx))
        trainX = X[cv_labels!=cv_idx, :, :]
        trainY = Y[cv_labels!=cv_idx, :]
        testX = X[cv_labels==cv_idx, :, :]
        testY = Y[cv_labels==cv_idx, :]

        print('Number of training data: {}'.format(trainX.shape[0]))
        print('Number of test data: {}'.format(testX.shape[0]))
        if outputfilename is not None:
            with open(outputfilename, 'a') as outputfile:
                outputfile.write('Number of training data: {}\n'.format(trainX.shape[0]))
                outputfile.write('Number of test data: {}\n'.format(testX.shape[0]))

        # Initializing Keras model
        print('Initializing Keras model...')
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        quantum_dmrg_model = QuantumKerasModel(dimvec, pos_label, nblabels, bond_len,
                                               optimizer=optimizer,
                                               nearzero_std=std)

        print(quantum_dmrg_model.summary())
        if outputfilename is not None:
            with open(outputfilename, 'a') as outputfile:
                model_summary = get_keras_model_summary(quantum_dmrg_model)
                outputfile.write(model_summary+'\n')

        # Training
        print('Training')
        quantum_dmrg_model.fit(trainX, trainY, batch_size=batch_size, epochs=nb_epochs)

        # Testing
        print('Testing')
        predictedY = quantum_dmrg_model.predict(testX)
        cross_entropy = - np.mean(np.sum(testY*np.log(predictedY), axis=1))
        print('Cross-entropy = {}'.format(cross_entropy))
        nbmatches = np.sum(np.argmax(testY, axis=1) == np.argmax(predictedY, axis=1))
        print('Number of matches = {}'.format(nbmatches))
        print('Accuracy = {:.2f}%'.format(nbmatches/nbdata*100))
        if outputfilename is not None:
            with open(outputfilename, 'a') as outputfile:
                outputfile.write('Cross-Validation Result\n')
                outputfile.write('Cross-entropy = {}\n'.format(cross_entropy))
                outputfile.write('Number of matches = {}\n'.format(nbmatches))
                outputfile.write('Accuracy = {:.2f}%\n'.format(nbmatches/nbdata*100))
                outputfile.write('\n')

        accuracies.append(nbmatches/nbdata)
        cv_time_records.append(time.time())

    endtime = time.time()

    # Timer calculation
    total_runtime = endtime - starttime
    dataread_time = endreaddata_time - starttime
    cv_times = [cv_time_records[i+1]-cv_time_records[i] for i in range(cv_fold)]
    times_strtoprint = 'Total runtime: {:.1f} sec\n'.format(total_runtime) + \
        'Time elapsed in data loading: {:.1f} sec\n'.format(dataread_time) + \
        '\n'.join(['\tTime for cross-validation {}: {:.1f} sec'.format(i, cv_times[i])
                   for i in range(cv_fold)]) + \
        '\n'

    print(times_strtoprint+'\n')
    print('Average accuracy = {:.2f}%'.format(np.mean(accuracies)*100))
    if outputfilename is not None:
        with open(outputfilename, 'a') as outputfile:
            outputfile.write('\nOverall Result\n')
            outputfile.write('===============\n')
            outputfile.write(hypparam_strtoprint)
            outputfile.write(times_strtoprint)
            outputfile.write('Average accuracy = {:.2f}%\n'.format(np.mean(accuracies) * 100))