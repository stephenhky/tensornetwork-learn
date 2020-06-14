
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
        self.construct_tensornetwork()

    def construct_tensornetwork(self):
        end_node = lambda i: tf.Variable(tf.random.uniform(shape=(2, self.m),
                                                           minval=0,
                                                           maxval=self.unihigh),
                                         name='mps_node_{}'.format(i),
                                         trainable=True)

        label_node = lambda i: tf.Variable(tf.random.uniform(shape=(2, self.m, self.m, self.nblabels),
                                                             minval=0,
                                                             maxval=self.unihigh),
                                           name='mps_node_{}'.format(i),
                                           trainable=True)
        normal_node = lambda i: tf.Variable(tf.random.uniform(shape=(2, self.m, self.m),
                                                              minval=0,
                                                              maxval=self.unihigh),
                                            name='mps_node_{}'.format(i),
                                            trainable=True)

        self.mps_tf_vars = [None] * self.dimvec
        for i in range(self.dimvec):
            self.mps_tf_vars[i] = tf.case(
                [(tf.math.logical_or(tf.math.equal(i, 0), tf.math.equal(i, self.dimvec - 1)), partial(end_node, i=i)),
                 (tf.math.equal(i, self.pos_label), partial(label_node, i=i))],
                default=partial(normal_node, i=i)
                )

        # model nodes
        self.nodes = [
            tn.Node(self.mps_tf_vars[i], name='node{}'.format(i), backend='tensorflow')
            for i in range(self.dimvec)
        ]

        # input nodes
        cosx = np.random.uniform(size=self.dimvec)
        self.input_nodes = [
            tn.Node(np.array([cosx[i], np.sqrt(1 - cosx[i] * cosx[i])]), name='input{}'.format(i), backend='tensorflow')
            for i in range(self.dimvec)]

    @tf.function
    def infer_single_datum(self, input):
        for i in range(self.dimvec):
            self.input_nodes[i].tensor = input[i, :]
        edges = [self.nodes[0][1] ^ self.nodes[1][1]]
        for i in range(1, self.dimvec - 1):
            edges.append(self.nodes[i][2] ^ self.nodes[i + 1][1])

        input_edges = [self.nodes[i][0] ^ self.input_nodes[i][0] for i in range(self.dimvec)]

        final_node = tn.contractors.greedy(self.nodes + self.input_nodes,
                                           output_edge_order=[self.nodes[self.pos_label][3]])
        # final_node = self.nodes[0] @ self.nodes[1]
        # for node in self.nodes[2:]+self.input_nodes:
        #     final_node = final_node @ node

        return final_node.tensor

    def call(self, inputs):
        return tf.vectorized_map(self.infer_single_datum, inputs)



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
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
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