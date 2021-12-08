import torch
import numpy as np

X = torch.tensor([1,2,3,4,5],dtype=torch.float32)
Y = torch.tensor([3,6,9,12,15],dtype=torch.float32)

W = torch.zeros(1,dtype=torch.float32 ,requires_grad=True)

def forward(X):
    return X*W


print(f'result before traning f(20):{forward(20)}')


def loss(y_hat,Y):
    return ((y_hat-Y)**2).mean()

learning_rate = 0.001
n_iters = 1000

for epoch in range(n_iters):

    y_hat = forward(X)
    l = loss(y_hat,Y)

    l.backward()

    with torch.no_grad():

        W -= learning_rate* W.grad

    W.grad.zero_()    

    if epoch % 10 ==0:
        print(f' epoch: {epoch+1}, W: {W} , loss: {l}')

print(f'Prediction after training: f(20)= {forward(20)}')

    
