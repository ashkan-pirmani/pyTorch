import torch
import torch.nn as nn


X = torch.tensor([[1],[2],[3],[4],[5],[6],[7],[8],[9]] , dtype=torch.float32)
Y = torch.tensor([[10],[20],[30],[40],[50],[60],[70],[80],[90]], dtype=torch.float32)

n_samples, n_features = X.size()
print(n_samples, n_features)

input_size = n_features
output_size = n_features

model = nn.Linear(input_size, output_size)

print(f'result before training f(3.2): {model(torch.tensor([3.2]))}')



learning_rate = 0.005
n_iters = 1500

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters() , lr=learning_rate)


for epoch in range(n_iters):

    y_hat = model(X)

    l = loss(y_hat,Y)

    l.backward()

    optimizer.step()

    optimizer.zero_grad()

    if epoch % 100 == 0:
        [w,b] = model.parameters()
        print(f'epoch {epoch+1}, w={w[0][0].item()} loss={l}')



print(f'result after training f(3.2): {model(torch.tensor([3.2]))}')

