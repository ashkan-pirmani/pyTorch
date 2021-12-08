import numpy as np


x = np.array([1,2,3,4], dtype=np.float32)
y = np.array([2,4,6,8], dtype=np.float32)


w = 0.0



def forward(x):
    return w*x

def loss(y,y_hat):
    return ((y_hat-y)**2).mean()

#MSE = 1/N * ((w*x)-y)**2)

def gradient(x,y,y_hat):
    return np.dot(2*x,y_hat-y).mean()


print(f'Prediction before training: f(5)= {forward(5)}')


learning_rate = 0.01
n_iters = 100

for epoch in range(n_iters):
    y_hat = forward(x)

    l = loss(y_hat, y)

    g= gradient(x,y,y_hat)


    w -= learning_rate*g

    if epoch % 1 == 0:
        print(f'epoch {epoch+1}: w={w}, loss={l}')

print(f'Prediction after training: f(5)= {forward(5)}')
