import torch 

X = torch.tensor([1,2,3,4,5])
Y = torch.tensor([2,4,6,8,10])

w = torch.zeros(1,dtype=torch.float32,requires_grad = True)

def forward(X):
    return X*w


print(f'result before training f(7): {forward(torch.tensor([7]))}')



def loss(y_hat,Y):
    return ((y_hat-Y)**2).mean()


learning_rate = 0.001
n_iters = 1000


for epoch in range(n_iters):

    y_hat = forward(X)

    l = loss(y_hat,Y)

    l.backward()

    with torch.no_grad():

        w -= learning_rate * w.grad


    w.grad.zero_()
    if epoch % 1 ==0:
        print(f'epoch {epoch+1}, w ={w} , loss={l}')

print(f'result after trainign f(7): {forward(torch.tensor([7]))}') 