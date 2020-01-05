import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms

train_dataset = dsets.MNIST(root=".", train=True, transform=transforms.ToTensor())
test_dataset = dsets.MNIST(root=".", train=False, transform=transforms.ToTensor())

batch_size = 100

num_iters = 3000

num_epochs = int(num_iters / (len(train_dataset) / batch_size))

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self):
        super(ConvolutionalNeuralNetwork, self).__init__()
        # Convolution 1
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, stride=1, padding=0, kernel_size=5)
        self.relu1 = nn.ReLU()

        # Maxpool
        #self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.avgpool1 = nn.AvgPool2d(kernel_size=2)
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=0, stride=1)
        self.relu2 = nn.ReLU()

        #self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.avgpool2 = nn.AvgPool2d(kernel_size=2)

        # fully connected layer
        #when padding = 2
        #self.fc1 = nn.Linear(32 * 7 * 7, 10)

        #when padding = 0
        self.fc1 = nn.Linear(32 * 4 * 4, 10)


    def forward(self, x):
        out = self.cnn1(x)
        out = self.relu1(out)
        #out = self.maxpool1(out)
        out = self.avgpool1(out)
        out = self.cnn2(out)
        out = self.relu2(out)
        #out = self.maxpool2(out)
        out = self.avgpool2(out)
        # reshaping because current matrix is (100,32,7,7) and we need (100, 32 * 7 * 7) for the linear layer
        # 100 is the batch size
        out = out.view(out.size(0), -1)
        out = self.fc1(out)

        return out


model = ConvolutionalNeuralNetwork()
criterion = nn.CrossEntropyLoss()

learning_rate = 0.1

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

#Parameter sizes
#Length of number of parameter sets
print(len(list(model.parameters())))
#16 kernels, 1 input channel, kernel size =  5*5
print(list(model.parameters())[0].size())
#16 biases
print(list(model.parameters())[1].size())
#32 kernels, 16 input size, 5* 5 kernels
print(list(model.parameters())[2].size())
#32 biases
print(list(model.parameters())[3].size())
# size for linear layer -> 32 * 7 *7 and 10 classes
print(list(model.parameters())[4].size())
#10 biases
print(list(model.parameters())[5].size())

iter = 0
for i in range(num_epochs):
    for (images, labels) in train_loader:
        # FFN had input (1, 28*28)
        # CNN has input (1,28,28) so no need to reshape
        # it accepts 2D inputs
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
                outputs = model(images)
                predicted = torch.max(outputs.data, 1)[1]
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            accuracy = (correct / total) * 100

            print("Iteration: {}, Loss: {}, Accuracy: {}".format(iter, loss.item(), accuracy))
