import torch 
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

X_numpy , y_numpy = datasets.make_regression(n_samples=150,n_features=1,noise=10,random_state=313)

X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))
y = y.view(y.shape[0], 1)


n_samples, n_features = X.shape


input_size = n_features
output_size = 1
learning_rate = 0.01

model = nn.Linear(input_size, output_size)
n_iters = 1000
learning_rate = 0.01


loss = nn.MSELoss()

optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

for epoch in range(n_iters):

    y_hat = model(X)

    l = loss(y_hat,y)

    optimizer.step()

    optimizer.zero_grad()

    if epoch % 100 == 0:
        print(f'epoch: {epoch+1}, loss: {l}')


predicted = model(X).detach().numpy()

plt.plot(X_numpy, y_numpy,'ro')
plt.plot(X_numpy, predicted, 'b')
plt.show()
