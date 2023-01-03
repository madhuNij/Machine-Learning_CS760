import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


class NeuralNetwork:
    def __init__(self, input=2, output=1, hidden=300, batch_size=20, epochs=100, lr=0.3):
        self.input = input
        self.output = output
        self.hidden = hidden
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.criterion = nn.BCEWithLogitsLoss()
        self.model = nn.Sequential(
        nn.Linear(self.input, self.hidden, bias=True),
        nn.Sigmoid(),
        nn.Linear(self.hidden, self.output, bias=True)
        )
        self.opt = optim.SGD(self.model.parameters(), lr=self.lr)

    def train_model(self, X, y):
        X = torch.Tensor(X)
        y = torch.Tensor(y)
        running_loss = 0.0
        correct = 0.0
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        for e in range(self.epochs):
            for i, data in enumerate(dataloader, 0):
                inputs, labels = data
                self.opt.zero_grad()
                outputs = self.model(inputs).squeeze()
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.opt.step()
                running_loss += loss.item()
                #correct = correct + torch.sum(outputs.argmax(1) == labels)


    def predict(self, X):
        X = torch.Tensor(X)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X)
            predictions = torch.round(torch.sigmoid(logits)).squeeze().detach().numpy()
        return predictions


'''
def get_data_loader(training = True):
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    train_set = torchvision.datasets.MNIST('./ data', train = True, download = True, transform = transform)
    test_set = torchvision.datasets.MNIST('./ data', train = False, transform = transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=32)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=32)
    if training:
        return train_loader
    else:
        return test_loader


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.normal_(m.weight)


def build_model():
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 200),
        nn.Sigmoid(),
        nn.Linear(200, 10)
    )
    #model.apply(init_weights)
    return model


def train_model(model, train_loader, criterion, epoch_train_error):
    model = model.train()
    opt = optim.SGD(model.parameters(), lr=0.001)
    running_loss = 0.0
    correct = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        opt.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        opt.step()
        running_loss += loss.item()
        correct = correct + torch.sum(outputs.argmax(1) == labels)
    epoch_train_error.append(float(1 - (correct / len(train_loader.dataset))))
    accuracy = (100. * correct / len(train_loader.dataset))
    print(f'Accuracy: {int(correct)}/{len(train_loader.dataset)} ({float(accuracy):.2f}%) Loss: {running_loss/len(train_loader):.3f}')


def test_model(model, test_loader, criterion, epoch_test_error):
    model = model.eval()
    correct = 0
    total = 0
    running_loss = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            running_loss += loss.item()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    epoch_test_error.append(1 - (correct/total))

    print(f'Accuracy: {100 * correct / total:.2f}%')


if __name__ == '__main__':
    epoch_test_error = []
    epoch_train_error = []
    criterion = nn.CrossEntropyLoss()
    train_loader = get_data_loader()
    test_loader = get_data_loader(False)
    model = build_model()
    opt = optim.SGD(model.parameters(), lr=0.001)
    epochs = 50
    for i in range(epochs):
        print("Epoch:", i)
        train_model(model, train_loader, criterion, epoch_train_error)
        test_model(model, test_loader, criterion, epoch_test_error)
    print(epoch_train_error)
    print(epoch_test_error)
    print("Average test error:", math.fsum(epoch_test_error)/epochs)
    epoch = np.arange(0, epochs)
    plt.plot(epoch, epoch_train_error, label='Training Error')
    plt.plot(epoch, epoch_test_error, label='Test Error')
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.legend(loc="upper right")
    plt.show()
'''