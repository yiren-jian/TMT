from torch.utils.data.dataset import Dataset

import os
import torch
import fnmatch
import numpy as np


class NYUv2(Dataset):
    """
    This file is directly modified from https://pytorch.org/docs/stable/torchvision/datasets.html
    """
    def __init__(self, root, train=True, return_idx=False):
        self.train = train
        self.root = os.path.expanduser(root)

        # Read the data file
        if train:
            self.data_path = root + '/train'
        else:
            self.data_path = root + '/val'

        # calculate data length
        self.data_len = len(fnmatch.filter(os.listdir(self.data_path + '/image'), '*.npy'))
        self.return_idx = return_idx

    def __getitem__(self, index):
        image = torch.from_numpy(np.moveaxis(np.load(self.data_path + '/image/{:d}.npy'.format(index)), -1, 0))
        semantic = torch.from_numpy(np.load(self.data_path + '/label/{:d}.npy'.format(index)))
        depth = torch.from_numpy(np.moveaxis(np.load(self.data_path + '/depth/{:d}.npy'.format(index)), -1, 0))
        normal = torch.from_numpy(np.moveaxis(np.load(self.data_path + '/normal/{:d}.npy'.format(index)), -1, 0))

        if self.return_idx:
            return image.type(torch.FloatTensor), semantic.type(torch.FloatTensor), depth.type(torch.FloatTensor), normal.type(torch.FloatTensor), index
        else:
            return image.type(torch.FloatTensor), semantic.type(torch.FloatTensor), depth.type(torch.FloatTensor), normal.type(torch.FloatTensor)

    def __len__(self):
        return self.data_len
