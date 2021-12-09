import torch
import torchvision
from torch.utils.data import Dataset
import numpy as np


class WineDataset(Dataset):

    def __init__(self , transform=None):
        xy = np.loadtxt('./wine.csv',dtype=np.float32,delimiter=',',skiprows=1)
        self.n_samples = xy.shape[0]

        self.x = xy[:,1:]
        self.y = xy[:,[0]]

        self.transform = transform


    def __getitem__(self, index):
        samples = self.x[index] , self.y[index]
        if self.transform is not None:
            samples = self.transform(samples)
            return samples

    def __len__(self):
        return self.n_samples


class ToTensor:
    def __call__(self, samples):
        inputs , labels = samples
        return torch.from_numpy(inputs), torch.from_numpy(labels)



dataset = WineDataset(transform=ToTensor())





