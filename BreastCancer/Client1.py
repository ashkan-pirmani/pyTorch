##########################################
#### Breast Cancer Classification
##########################################

import torch
import numpy as np
import torch.nn as nn
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


##########################################

dataset = datasets.load_breast_cancer()

X,y = dataset.data , dataset.target

nX = X[0:320,:]
ny = y[0:320]

ny_samples, ny_features = nX.shape
print(ny_samples,ny_features)


#########################################


X_train, X_test ,y_train, y_test = train_test_split(nX,ny,test_size=0.2,random_state=313)


sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))

y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))


y_train = y_train.view(y_train.shape[0],1)
y_test = y_test.view(y_test.shape[0],1)


#########################################


class LogisticRegression(nn.Module):
    def __init__(self,n_input_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features,1)


    def forward(self,X):
        y_hat = torch.sigmoid(self.linear(X))
        return y_hat


##########################################


model = LogisticRegression(ny_features)

loss = nn.BCELoss()

optimizer = torch.optim.SGD(model.parameters(), lr=0.001)


n_iters = 10


for epoch in range(n_iters):

    y_hat = model(X_train)
    l = loss(y_hat, y_train)

    l.backward()

    optimizer.step()

    optimizer.zero_grad()

    if epoch % 1 == 0:
        print(f'epoch{epoch+1} , loss: {l}')


with torch.no_grad():

    y_hat = model(X_test)
    y_hat_cls = y_hat.round()

    accuracy = y_hat_cls.eq(y_test).sum() / float(y_test.shape[0])

    print(f'accuracy:{accuracy}')


torch.save(LogisticRegression, 'model_client1.pth')


checkpoint = {
    "epoch": epoch,
    "model_state" : model.state_dict(),
    "optimizer_state": optimizer.state_dict()
}

torch.save(checkpoint, "checkpoint_client1.pth")