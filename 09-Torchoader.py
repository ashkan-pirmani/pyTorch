import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math


class WineDataset(Dataset):

    def __init__(self):
        # Initialize data, download, etc.
        # read with numpy or pandas
        xy = np.loadtxt('./wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.n_samples = xy.shape[0]

        # here the first column is the class label, the rest are the features
        self.x_data = torch.from_numpy(xy[:, 1:]) # size [n_samples, n_features]
        self.y_data = torch.from_numpy(xy[:, [0]]) # size [n_samples, 1]

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples



dataset = WineDataset()

dataloader = DataLoader(dataset=dataset,batch_size=4 ,shuffle=True, num_workers=0 )


dataiter = iter(dataloader)
data = dataiter.next()
features , labels = data



n_epochs = 5
total_samples = len(dataset)
n_iters = math.ceil(total_samples / 4)



for epoch in range(n_epochs):
    for i ,(inputs, labels) in enumerate(dataloader):
        if (i+1) % 5== 0:
            print(f'Epoch: {epoch+1}/{n_epochs}, Step {i+1}/{n_iters}| Inputs {inputs.shape} | Labels {labels.shape}')
