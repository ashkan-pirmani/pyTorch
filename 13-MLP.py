import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


input_size = 784 # number28x28
hidden_size = 250
number_classes = 10
number_epochs = 2
batch_size = 64
learning_rate = 0.005


######MNIST #################################################################################################

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform= transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train = False, transform = transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset , batch_size = batch_size)





class MLP(nn.Module):
    def __init__(self,input_size,hidden_size,number_classes):
        super(MLP,self).__init__()
        self.l1 = nn.Linear(input_size,hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size,number_classes)
    
    def forward(self,x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out


model = MLP(input_size,hidden_size,number_classes)

loss = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)


n_total_steps = len(train_loader)


for epoch in range(number_epochs):
    for i,(images,labels) in enumerate(train_loader):
        images = images.reshape(-1, 28*28)


        y_hat = model(images)
        l = loss(y_hat,labels)


        optimizer.zero_grad()
        l.backward()

        optimizer.step()

        if i % 100 == 0:
            print(f'epoch {epoch+1}/{number_epochs}, step {i+1}/{n_total_steps}, loss = {l.item()}')



    
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28)

        y_hat = model(images)
        _, predictions = torch.max(y_hat,1)
        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()


accuracy = n_correct/n_samples

print(f'Accuracy:{accuracy}')




