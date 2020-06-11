
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

    def construct_tensornetwork(self):
        self.mps_tf_vars = [None] * self.dimvec
        for i in range(self.dimvec):
            if i == 0 or i == self.dimvec - 1:
                self.mps_tf_vars[i] = tf.Variable(tf.random.uniform(shape=(2, self.m),
                                                                    minval=0,
                                                                    maxval=self.unihigh),
                                                  name='mps_node_{}'.format(i),
                                                  trainable=True)
            elif i == self.pos_label:
                self.mps_tf_vars[i] = tf.Variable(tf.random.uniform(shape=(2, self.m, self.m, self.nblabels),
                                                                    minval=0,
                                                                    maxval=self.unihigh),
                                                  name='mps_node_{}'.format(i),
                                                  trainable=True)
            else:
                self.mps_tf_vars[i] = tf.Variable(tf.random.uniform(shape=(2, self.m, self.m),
                                                                    minval=0,
                                                                    maxval=self.unihigh),
                                                  name='mps_node_{}'.format(i),
                                                  trainable=True)

        # model nodes
        self.nodes = [
            tn.Node(self.mps_tf_vars[i], name='node{}'.format(i), backend='tensorflow')
            for i in range(self.dimvec)
        ]

    def infer_single_datum(self, input):
        input_nodes = [tn.Node(input[i]) for i in range(self.dimvec)]
        edges = [self.nodes[0][1] ^ self.nodes[1][1]]
        for i in range(1, self.dimvec - 1):
            edges.append(self.nodes[i][2] ^ self.nodes[i + 1][1])

        input_edges = [self.nodes[i][0] ^ input_nodes[i][0] for i in range(self.dimvec)]

        final_node = tn.contractors.auto(self.nodes + input_nodes,
                                         output_edge_order=[self.nodes[self.pos_label][3]])
        return final_node.tensor

    def call(self, inputs):
        return tf.vectorized_map(self.infer_single_datum, inputs)



class QuantumTensorNetworkClassifier(ExperimentalClassifier):
    def __init__(self, pos_label, m, unihigh=0.05):
        self.dimvec = None
        self.pos_label = pos_label
        self.nblabels = None
        self.m = m
        self.unihigh = unihigh

    def construct_tensornetwork(self):
        # model nodes
        self.nodes = [
            tn.Node(np.random.uniform(high=self.unihigh, size=(2, self.m)), name='node{}'.format(i))
                 if i == 0 or i == self.dimvec - 1
                 else tn.Node(np.random.uniform(high=self.unihigh, size=(2, self.m, self.m)), name='node{}'.format(i))
                 for i in range(self.dimvec)
        ]
        self.nodes[self.pos_label] = tn.Node(np.random.uniform(high=self.unihigh,
                                                               size=(2, self.m, self.m, self.nblabels)),
                                             name='label_node')

        # input nodes (initialization)
        cosx = np.random.uniform(size=self.dimvec)
        self.input_nodes = [
            tn.Node(np.array([cosx[i], np.sqrt(1 - cosx[i] * cosx[i])]), name='input{}'.format(i))
            for i in range(self.dimvec)
        ]

    def connect_edges(self):
        self.edges = [self.nodes[0][1] ^ self.nodes[1][1]]
        for i in range(1, self.dimvec - 1):
            self.edges.append(self.nodes[i][2] ^ self.nodes[i + 1][1])

        self.input_edges = [self.nodes[i][0] ^ self.input_nodes[i][0]
                            for i in range(self.dimvec)]

    def fit(self, X, Y, *args, **kwargs):
        self.dimvec = X.shape[1]
        self.nblabels = Y.shape[1]
        pass

    def fit_batch(self, dataset, *args, **kwargs):
        pass

    def predict_proba(self, X, *args, **kwargs):
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
        pass

    def persist(self, path):
        pass

    def trim(self):
        pass

    @classmethod
    def load(cls, path, *args, **kwargs):
        pass