import torch
import torch.nn as nn


X = torch.tensor([1,2,3,4,5,6,7,8,9] , dtype=torch.float32)
Y = torch.tensor([10,20,30,40,50,60,70,80,90], dtype=torch.float32)

W=torch.zeros(1,dtype=torch.float32,requires_grad=True)

def forward(X):
    return X*W

print(f'result before training f(3.2): {forward(torch.tensor([3.2]))}')




learning_rate = 0.001
n_iters = 1000

loss = nn.MSELoss()
optimizer = torch.optim.SGD([W] , lr=learning_rate)


for epoch in range(n_iters):

    y_hat = forward(X)

    l = loss(y_hat,Y)

    l.backward()

    optimizer.step()

    optimizer.zero_grad()

    if epoch % 100 == 0:
        print(f'epoch {epoch+1}, w={W}, loss={l}')



print(f'result after training f(3.2): {forward(torch.tensor([3.2]))}')

