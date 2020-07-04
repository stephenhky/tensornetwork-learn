
import io
import json
import argparse
from collections import defaultdict

import numpy as np
import tensornetwork as tn
import tensorflow as tf
from tqdm import tqdm
from scipy.sparse import dok_matrix


###### Keras Layer for Tensor Network ##########

class QuantumTNLayer(tf.keras.layers.Layer):
    def __init__(self, nboutputs, nearzero_std=1e-4):
        super(QuantumTNLayer, self).__init__()
        self.nboutputs = nboutputs
        self.nearzero_std = nearzero_std

    def build(self, input_shape):
        self.nbdata = input_shape[0]
        self.vecdim = input_shape[1]

        self.tr_var = tf.Variable(tf.random.normal((self.nboutputs, self.vecdim),
                                                   mean=0.0,
                                                   stddev=self.nearzero_std) + \
                                  tf.eye(self.nboutputs, num_columns=self.vecdim),
                                  name='tr_var',
                                  trainable=True)
        self.bias = tf.Variable(tf.random.normal((self.nboutputs,),
                                                 mean=0.0,
                                                 stddev=self.nearzero_std),
                                name='bias',
                                trainable=True)

    def infer_single_datum(self, input):
        input_node = tn.Node(input, backend='tensorflow')
        tr_node = tn.Node(self.tr_var, backend='tensorflow')
        edge = input_node[0] ^ tr_node[1]
        final_node = tn.contract(edge)
        return final_node.tensor + self.bias

    def call(self, X):
        Ypred = tf.vectorized_map(self.infer_single_datum, X)
        return Ypred


def get_tn_keras_model(fmap, lmap, learning_rate=1e-4):
    tn_model = tf.keras.Sequential([
        tf.keras.Input(shape=(len(fmap),)),
        QuantumTNLayer(len(lmap)),
        tf.keras.layers.Softmax()
    ])
    tn_model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                     loss=tf.keras.losses.CategoricalCrossentropy())
    return tn_model


# From: https://stackoverflow.com/questions/41665799/keras-model-summary-object-to-string
def get_keras_model_summary(model):
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\n'))
    summary_string = stream.getvalue()
    stream.close()
    return summary_string


###### Data processing ######

def parse_data(filepath):
    for line in open(filepath, 'r'):
        yield json.loads(line)


def get_feature_summary(alldata, features):
    summary = {}
    for feature in features['quantitative']:
        summary[feature] = {}
        summary[feature]['sum'] = 0.
        summary[feature]['sqsum'] = 0.
        summary[feature]['nbdata'] = 0
    for feature in features['binary']:
        summary[feature] = {}
        summary[feature][True] = 0
        summary[feature][False] = 0
    for feature in features['qualitative']:
        summary[feature] = defaultdict(lambda: 0)
        summary[features['target']] = defaultdict(lambda: 0)

    for datum in alldata:
        for feature in features['quantitative']:
            summary[feature]['sum'] += datum.get(feature, 0)
            summary[feature]['sqsum'] += datum.get(feature, 0) * datum.get(feature, 0)
            if datum.get(feature) is not None:
                summary[feature]['nbdata'] += 1
        for feature in features['binary']:
            if datum.get(feature) is not None and datum.get(feature) in [True, False]:
                summary[feature][datum.get(feature)] += 1
        for feature in features['qualitative']:
            val = datum.get(feature)
            if val is not None:
                summary[feature][val] += 1
        label = datum.get(features['target'])
        summary[features['target']][label] += 1

    for feature in features['quantitative']:
        summary[feature]['mean'] = summary[feature]['sum'] / summary[feature]['nbdata']
        summary[feature]['std'] = np.sqrt(
            summary[feature]['sqsum'] / summary[feature]['nbdata'] - summary[feature]['mean'] * summary[feature][
                'mean'])
    for feature in features['qualitative']:
        summary[feature] = dict(summary[feature])

    summary[features['target']] = dict(summary[features['target']])

    return summary


def transform_data_to_featurevector(data_iterator, features, feature_summary):
    alldata = [datum for datum in data_iterator]

    # mapping
    feature_map = {}
    label_map = {}
    for feature in features['quantitative']:
        feature_map[feature] = len(feature_map)
    for feature in features['binary']:
        feature_map['{}:True'.format(feature)] = len(feature_map)
        feature_map['{}:False'.format(feature)] = len(feature_map)
    for feature in features['qualitative']:
        for val in feature_summary[feature]:
            feature_map['{}:{}'.format(feature, val)] = len(feature_map)
    for val in feature_summary[features['target']]:
        label_map[val] = len(label_map)

    feature_matrix = dok_matrix((len(alldata), len(feature_map)))
    label_matrix = dok_matrix((len(alldata), len(label_map)))
    print('Parsing data')
    for i in tqdm(range(len(alldata))):
        datum = alldata[i]
        for feature in features['quantitative']:
            val = datum.get(feature)
            z_val = (val - feature_summary[feature]['mean']) / feature_summary[feature][
                'std'] if val is not None else 0.
            feature_matrix[i, feature_map[feature]] = z_val
        for feature in features['binary']:
            val = datum.get(feature)
            if val is not None:
                if val:
                    feature_matrix[i, feature_map['{}:True'.format(feature)]] = 1.
                else:
                    feature_matrix[i, feature_map['{}:False'.format(feature)]] = 1.
        for feature in features['qualitative']:
            val = datum.get(feature)
            if val is not None and val in feature_summary[feature].keys():
                feature_matrix[i, feature_map['{}:{}'.format(feature, val)]] = 1.

        label = datum.get(features['target'])
        if label is not None and label in feature_summary[features['target']]:
            label_matrix[i, label_map[label]] = 1.

    return feature_matrix, label_matrix, feature_map, label_map


######### Arguement Parsing ###########

def get_argparser():
    argparser = argparse.ArgumentParser(description='fruit experiments')
    argparser.add_argument('filepath', help='path of the fruits data')
    argparser.add_argument('--featurefilepath', default='fruit_features.json', help='feature file path')
    argparser.add_argument('--cvfold', default=5, type=int, help='number of cross-validations')
    argparser.add_argument('--learning_rate', default=1e-4, type=float, help='learning rate of Adam optimizer')
    argparser.add_argument('--nbepochs', default=100, type=int, help='number of epochs')
    argparser.add_argument('--batch_size', default=100, type=int, help='batch size')
    argparser.add_argument('--output_file', default=None, help='output log')
    return argparser


if __name__ == '__main__':
    argparser = get_argparser()
    args = argparser.parse_args()

    # print hyperparameters
    cvfold = args.cvfold
    learning_rate = args.learning_rate
    nbepochs = args.nbepochs
    batch_size = args.batch_size
    outputfilename = args.output_file
    hypparam_strtoprint = 'Filepath: {}\n'.format(args.filepath) + \
        'Number of cross-validations: {}\n'.format(cvfold) + \
        'Learning rate for Adam optimizer: {}\n'.format(learning_rate) + \
        'Number of epochs: {}\n'.format(nbepochs) + \
        'Batch size: {}\n'.format(batch_size)
    if outputfilename is not None:
        outputfile = open(outputfilename, 'w')
        outputfile.write(hypparam_strtoprint)
        outputfile.close()

    # data processing
    features = json.load(open(args.featurefilepath, "r"))
    fruit_data_iterator = parse_data(args.filepath)
    summary = get_feature_summary(fruit_data_iterator, features)
    fruit_data_iterator = parse_data(args.filepath)
    X, Y, fmap, lmap = transform_data_to_featurevector(fruit_data_iterator, features, summary)
    nbdata = Y.shape[0]
    assert X.shape[0] == Y.shape[0]

    print('Features')
    print(features)
    print(lmap)
    print('Number of data: {}'.format(nbdata))
    if outputfilename is not None:
        with open(outputfilename, 'a') as outputfile:
            outputfile.write('Features\n')
            outputfile.write(str(features))
            outputfile.write('\n')
            outputfile.write('Feature summary\n')
            outputfile.write(str(summary))
            outputfile.write('\n')
            outputfile.write('Labels\n')
            outputfile.write(str(lmap))
            outputfile.write('\n')
            outputfile.write('Number of data: {}\n'.format(nbdata))

    # cross-validation
    print('Cross-Validation')
    print('================')
    if outputfilename is not None:
        with open(outputfilename, 'a') as outputfile:
            outputfile.write('Cross-Validation\n')
            outputfile.write('================\n')
    accuracies = []
    cv_labels = np.random.choice(range(cvfold), size=nbdata)
    for cv_label in range(cvfold):
        # training
        trainX = X[cv_labels!=cv_label, :]
        trainY = Y[cv_labels!=cv_label, :]
        assert trainX.shape[0] == trainY.shape[0]
        print('Number of training data: {}'.format(trainY.shape[0]))
        if outputfilename is not None:
            with open(outputfilename, 'a') as outputfile:
                outputfile.write('Number of training data: {}\n'.format(trainY.shape[0]))

        tn_model = get_tn_keras_model(fmap, lmap, learning_rate=learning_rate)
        print(tn_model.summary())
        if outputfilename is not None:
            with open(outputfilename, 'a') as outputfile:
                model_summary = get_keras_model_summary(tn_model)
                outputfile.write(model_summary+'\n')

        tn_model.fit(trainX.toarray(), trainY.toarray(), epochs=nbepochs, batch_size=batch_size)

        # test
        testX = X[cv_labels==cv_label, :]
        testY = Y[cv_labels==cv_label, :]
        assert testX.shape[0] == testY.shape[0]
        print('Number of test data: {}'.format(testY.shape[0]))
        predY = tn_model.predict_proba(testX.toarray())
        cross_entropy = - np.mean(np.sum(testY.toarray()*np.log(predY), axis=1))
        print('Cross-entropy = {}'.format(cross_entropy))
        nbmatches = np.sum(np.argmax(testY.toarray(), axis=1) == np.argmax(predY, axis=1))
        print('Number of matches = {}'.format(nbmatches))
        accuracy = nbmatches / testY.shape[0]
        print('Accuracy = {:.2f}%'.format(accuracy * 100))
        accuracies.append(accuracy)

        if outputfilename is not None:
            with open(outputfilename, 'a') as outputfile:
                outputfile.write('Number of test data: {}\n'.format(testY.shape[0]))
                outputfile.write('Cross-entropy = {}\n'.format(cross_entropy))
                outputfile.write('Number of matches = {}\n'.format(nbmatches))
                outputfile.write('Accuracy = {:.2f}%\n'.format(accuracy * 100))

    print('\nFinal Results')
    print('--------------')
    print('Average accuracy = {:.2f}%'.format(np.mean(accuracies)*100))
    if outputfilename is not None:
        with open(outputfilename, 'a') as outputfile:
            outputfile.write('\nFinal Results\n')
            outputfile.write('--------------\n')
            outputfile.write('Average accuracy = {:.2f}%\n'.format(np.mean(accuracies) * 100))
