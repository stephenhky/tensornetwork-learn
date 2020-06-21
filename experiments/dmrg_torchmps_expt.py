
import json

import numpy as np
import numba
import torch
import torchmps
from tqdm import tqdm

@numba.njit(numba.float64[:, :](numba.float64[:]))
def convert_pixels_to_tnvector(pixels):
    tnvector = np.concatenate(
        (np.expand_dims(np.cos(0.5*np.pi*pixels/256.), axis=0),
         np.expand_dims(np.sin(0.5*np.pi*pixels/256.), axis=0)),
        axis=0
    ).T
    return tnvector


class MNISTJSON_Dataset(torch.utils.data.Dataset):
    def __init__(self, filepath, cvidx):
        self.filepath = filepath
        self.cvidx = cvidx

        self.digit_map = {digit: int(digit) for digit in '0123456789'}
        self.X = None
        self.Y = None

        for pixels, digit in self.generate_data(open(filepath, 'r')):
            pixel_vector = convert_pixels_to_tnvector(np.array(pixels))
            ans = np.zeros((10,))
            ans[self.digit_map[digit]] = 1.
            if self.X is None:
                self.X = np.expand_dims(pixel_vector, axis=0)
                self.Y = np.expand_dims(ans, axis=0)
            else:
                self.X = np.concatenate((self.X,
                                         np.expand_dims(pixel_vector, axis=0)),
                                        axis=0)
                self.Y = np.concatenate((self.Y,
                                         np.expand_dims(ans, axis=0)),
                                        axis=0)

        assert self.X.shape[0] == len(self.cvidx)
        self.change_subset(0, 'train')

    def change_subset(self, current_fold, train_or_test):
        assert train_or_test in ['train', 'test']

        self.current_fold = current_fold
        self.train_or_test = train_or_test

        if self.train_or_test == 'test':
            self.section_rowid = np.arange(self.X.shape[0])[self.cvidx==current_fold]
        else:
            self.section_rowid = np.arange(self.X.shape[0])[self.cvidx!=current_fold]

    def generate_data(self, mnist_file):
        for line in mnist_file:
            data = json.loads(line)
            pixels = np.array(data['pixels'])
            digit = data['digit']
            yield pixels, digit

    def __getitem__(self, idx):
        assert idx <= len(self.cvidx)
        rowidx = self.cvidx[idx]
        x = torch.Tensor(self.X[rowidx, :, :])
        y = torch.Tensor(self.Y[rowidx, :])
        return x, y

    def __len__(self):
        return len(self.cvidx)


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
    learning_rate = 1e-4

    # Prepare for cross-validation
    cv_labels = np.random.choice(range(cv_fold), size=nbdata)
    dataset = MNISTJSON_Dataset('mnist_784/mnist_784.json', cv_labels)

    # cross-validation
    accuracies = []
    for cv_idx in range(cv_fold):
        print('Round {}'.format(cv_idx))
        dataset.change_subset(cv_idx, 'train')

        mps = torchmps.torchmps.MPS(input_dim=dimvec, output_dim=nblabels, bond_dim=bond_len,
                                    adaptive_mode=False, periodic_bc=False, label_site=pos_label)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(mps.parameters(), lr=learning_rate)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

        for _ in tqdm(range(nb_epochs)):
            for i, (x_train, y_train) in enumerate(dataloader):
                optimizer.zero_grad()
                target_indices = torch.max(y_train, 1)[1]
                y_pred = mps(x_train)
                loss = criterion(y_pred, target_indices)
                loss.backward()
                optimizer.step()

        dataset.change_subset(cv_idx, 'test')
        test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        nb_matches = 0
        for x_test, y_test in enumerate(test_dataloader):
            y_test_indices = torch.max(y_test, 1)[1]
            y_test_pred = mps(x_test)
            y_pred_indices = torch.max(y_test_pred, 1)[1]
            nb_matches += int(torch.sum(y_test_indices == y_pred_indices))

        print('Number of test data = {}'.format(len(dataset)))
        print('Number of matches = {}'.format(nb_matches))
        accuracy = nb_matches / len(dataset)
        print('Accuracy = {:.2f}%'.format(accuracy*100))

        accuracies.append(accuracy)

    print('Average accuracies = {:.2f}%'.format(np.mean(accuracies)*100))
