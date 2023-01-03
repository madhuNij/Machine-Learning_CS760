import time
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
from torchvision import datasets, transforms

class Net(nn.Module):
    def __init__(self, epochs = 5):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(784, 300)
        self.linear2 = nn.Linear(300, 10)

        self.epochs = epochs

    def forward_pass(self, x):
        x = self.linear1(x)
        x = torch.sigmoid(x)
        x = self.linear2(x)
        x = torch.softmax(x, dim=0)
        return x

    def one_hot_encode(self, y):
        encoded = torch.zeros([10], dtype=torch.float64)
        encoded[y[0]] = 1.
        return encoded

    def train(self, train_loader, optimizer, criterion):
        start_time = time.time()
        loss = None
        correct = 0.0
        for iteration in range(self.epochs):
            for x, y in train_loader:
                y = self.one_hot_encode(y)
                optimizer.zero_grad()
                output = self.forward_pass(torch.flatten(x))
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
                correct = correct + torch.sum(output == y)
            accuracy = (100. * correct / len(train_loader.dataset))
            print("Accuracy:", accuracy)
            print('Epoch: {0}, Time Spent: {1:.2f}s, Loss: {2}'.format(
                iteration + 1, time.time() - start_time, loss
            ))

transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True, transform=transform))

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, transform=transform))
model = Net()

optimizer = optim.SGD(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

model.train(train_loader, optimizer, criterion)