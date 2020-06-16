
from functools import partial

import numpy as np
import tensorflow as tf
import tensornetwork as tn
from mlexpt.ml.core import ExperimentalClassifier

# Note: input data has already been trigonometrized.

class QuantumDMRGLayer(tf.keras.layers.Layer):
    def __init__(self, dimvec, pos_label, nblabels, bond_len, unihigh):
        super(QuantumDMRGLayer, self).__init__()
        self.dimvec = dimvec
        self.pos_label = pos_label
        self.nblabels = nblabels
        self.m = bond_len
        self.unihigh = unihigh

        self.mps_tensors = [tf.Variable(tf.random.uniform(shape=self.mps_tensor_shape(i),
                                                          minval=0,
                                                          maxval=self.unihigh),
                                        trainable=True,
                                        name='mps_tensors_{}'.format(i))
                            for i in range(self.dimvec)]

    def mps_tensor_shape(self, idx):
        if idx == 0 or idx == self.dimvec - 1:
            return (2, self.dimvec)
        elif idx == self.pos_label:
            return (2, self.dimvec, self.dimvec, self.nblabels)
        else:
            return (2, self.dimvec, self.dimvec)

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


class QuantumTensorNetworkClassifier(ExperimentalClassifier):
    def __init__(self, pos_label, m, unihigh=0.05, optimizer='adam'):
        self.dimvec = None
        self.pos_label = pos_label
        self.nblabels = None
        self.m = m
        self.unihigh = unihigh
        self.optimizer = optimizer

        self.trained = False

    def fit(self, X, Y, *args, **kwargs):
        self.dimvec = X.shape[1]
        self.nblabels = Y.shape[1]

        self.quantum_dmrg_model = tf.keras.Sequential([
            tf.keras.Input(shape=(self.dimvec, 2)),
            QuantumDMRGLayer(dimvec=self.dimvec,
                             pos_label=self.pos_label,
                             nblabels=self.nblabels,
                             bond_len=self.m,
                             unihigh=self.unihigh),
            tf.keras.layers.Softmax()
        ])
        loss_fn = tf.keras.losses.CategoricalCrossentropy()
        self.quantum_dmrg_model.compile(optimizer=self.optimizer, loss=loss_fn)
        self.quantum_dmrg_model.fit(X, Y)

    def fit_batch(self, dataset, *args, **kwargs):
        X = None
        Y = None
        for x_batch, y_batch in dataset:
            X = x_batch if X is None else np.append(X, x_batch, axis=0)
            Y = y_batch if Y is None else np.append(Y, y_batch, axis=0)
        self.fit(X, Y, *args, **kwargs)

    def predict_proba(self, X, *args, **kwargs):
        if not self.trained:
            raise Exception('Model not trained!')

        assert X.shape[1] == self.dimvec

        nbdata = X.shape[0]
        predy = np.zeros((nbdata, self.nblabels))
        for i in range(nbdata):
            # put input data to nodes
            for j in range(self.dimvec):
                self.input_nodes[j].tensor = X[i, j, :]

            # connect the edges
            self.connect_edges()

            # computation
            final_node = tn.contractors.auto(self.nodes + self.input_nodes,
                                             output_edge_order=[self.nodes[self.pos_label][3]])
            predy[i, :] = final_node.tensor

        return predy

    def predict_proba_batch(self, dataset, *args, **kwargs):
        X = None
        for x_batch, _ in dataset:
            X = x_batch if X is None else np.append(X, x_batch, axis=0)
        return self.predict_proba(X, *args, **kwargs)

    def persist(self, path):
        pass

    def trim(self):
        pass

    @classmethod
    def load(cls, path, *args, **kwargs):
        pass