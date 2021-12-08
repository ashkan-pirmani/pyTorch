import numpy as np


X = np.array([5,4,3,2,1],dtype=np.float32)
Y = np.array([20,16,12,8,4],dtype=np.float32)

w = 0

def forward(X):
    return X*w


print(f'result before training for f(10): {forward(10)}')




def loss(y_hat,Y):
    return ((y_hat-Y)**2).mean()

def gradient(y_hat,Y,X):
    return np.dot(2*X,y_hat-Y).mean()


learning_rate = 0.001

n_iters = 100

for epoch in range(n_iters):

    y_hat = forward(X)
    l = loss(y_hat,Y)
    g = gradient(y_hat,Y,X)

    w -= learning_rate*g

    if epoch % 1 == 0:
        print(f'epoch :{epoch+1}, w :{w}, loss:{l}')

print(f'result after training for f(10): {forward(10)}')


