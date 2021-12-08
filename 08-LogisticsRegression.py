import torch
import torch.nn as nn
from sklearn import datasets
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


dataset = datasets.load_breast_cancer()

X,y = dataset.data, dataset.target

n_samples , n_feature = X.shape

X_train, X_test ,y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=313)


sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))

y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))


y_train = y_train.view(y_train.shape[0],1)
y_test = y_test.view(y_test.shape[0],1)


#########################################################

class LogisticRegression(nn.Module):
    def __init__(self,n_input_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features,1)


    def forward(self,X):
        y_hat = torch.sigmoid(self.linear(X))
        return y_hat


model = LogisticRegression(n_feature)

loss = nn.BCELoss()

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


n_iters = 1000


for epoch in range(n_iters):

    y_hat = model(X_train)
    l = loss(y_hat, y_train)

    l.backward()

    optimizer.step()

    optimizer.zero_grad()

    if epoch % 100 == 0:
        print(f'epoch{epoch+1} , loss: {l}')


with torch.no_grad():

    y_hat = model(X_test)
    y_hat_cls = y_hat.round()

    accuracy = y_hat_cls.eq(y_test).sum() / float(y_test.shape[0])

    print(f'accuracy:{accuracy}')

