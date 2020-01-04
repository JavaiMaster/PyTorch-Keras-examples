import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms

train_dataset = dsets.MNIST(root="./data", train=True, transform=transforms.ToTensor())
test_dataset = dsets.MNIST(root="./data", train=False, transform=transforms.ToTensor())

batch_size = 100

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

num_iter = 3000
num_epochs = int(num_iter / (len(train_dataset) / batch_size))


class FeedForwardNeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_units, num_classes):
        super(FeedForwardNeuralNetwork, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_units)
        #self.sigmoid = nn.Sigmoid()
        #self.sigmoid = nn.Tanh()
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(hidden_units, hidden_units)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(hidden_units, num_classes)

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu1(out)
        out = self.linear2(out)
        out = self.relu2(out)
        out = self.linear3(out)
        return out


# same as logistc regression
input_dim = 28 * 28
hidden_dim = 100
output_dim = 10

model = FeedForwardNeuralNetwork(input_dim, hidden_dim, output_dim)

criterion = nn.CrossEntropyLoss()

learning_rate = 0.1

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Model sizes
print(list(model.parameters())[0].size())
print(list(model.parameters())[1].size())
print(list(model.parameters())[2].size())
print(list(model.parameters())[3].size())
print(list(model.parameters())[4].size())
print(list(model.parameters())[5].size())

iter = 0
for i in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, 28 * 28)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        iter += 1
        if iter % 500 == 0:
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.reshape(-1, 28 * 28)
                outputs = model(images)
                predicted = torch.max(outputs.data, 1)[1]
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            accuracy = (correct / total) * 100

            print("Iteration: {}, Loss: {}, Accuracy: {}".format(iter, loss.item(), accuracy))
