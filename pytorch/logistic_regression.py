import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import numpy as np
import matplotlib.pyplot as plt

# add download=True for first time
train_dataset = dsets.MNIST(root="./data", train=True, transform=transforms.ToTensor())

print(len(train_dataset))
print(type(train_dataset[0]))
print(train_dataset[29][1])

# showing the dataset image using pyplot
# show_img = train_dataset[29][0].numpy().reshape(28, 28)

# plt.imshow(show_img, cmap="gray")
# plt.show()

test_dataset = dsets.MNIST(root="./data", train=False, transform=transforms.ToTensor())
print(len(test_dataset))

# Making data iterable

batch_size = 100

# 1 epoch is one pass through entire data
n_iterations = 3000
num_epochs = int(n_iterations / (len(train_dataset) / batch_size))
print(num_epochs)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        output = self.linear(x)
        return output


# input image size is 28*28
# output dimn is 10 which is 0,1,2,34,5,6,7,8,9

input_size = 28 * 28
output_size = 10

model = LogisticRegressionModel(input_size, output_size)

# nn.CrossEntropyLoss takes care of softmax and cross entropy loss
criterion = nn.CrossEntropyLoss()

learning_rate = 0.001
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# parametr sizes (10,784) for weights
print(list(model.parameters())[0].size())
# parameter sizes 10 for biases
print(list(model.parameters())[1].size())

iter = 0
for epoch in range(num_epochs):
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
                test_images = images.reshape(-1, 28 * 28)
                outputs = model(test_images)
                predicted = torch.max(outputs.data, 1)[1]
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            accuracy = 100 * (correct / total)

            print("Iteration: {}, Loss: {}, Accuracy: {}".format(iter, loss.item(), accuracy))

