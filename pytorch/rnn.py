import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms

train_dataset = dsets.MNIST(root="./data", train=True, transform=transforms.ToTensor())
test_dataset = dsets.MNIST(root="./data", train=False, transform=transforms.ToTensor())

batch_size = 100

num_iters = 3000

num_epochs = int(num_iters / (len(train_dataset) / batch_size))

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


class RecurrentNeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(RecurrentNeuralNetwork, self).__init__()
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim
        # batch_first = True causes input/output dimension to have shape as (batch_dim, seq_dim (which is the hidden_dim) , input_dim)
        self.rnn1 = nn.RNN(input_size=input_dim, hidden_size=hidden_dim, num_layers=layer_dim, batch_first=True,
                          nonlinearity='tanh')
        # Use tanh to avoid exploding gradients which makes loss become NaN
        # self.rnn2 = nn.RNN(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=layer_dim, batch_first=True,
        #                    nonlinearity='relu')

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # This is of size (layer_dim, batch_size, hidden_dim)
        # This is the fist hidden layer
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)

        # We get back the final hidden layer and output
        out, hn = self.rnn1(x, h0)

        #out, hn = self.rnn2(out, h0)

        # out.size - > (100,28,100)
        # Out of which we only need output from last time step hence the -1

        out = self.fc(out[:, -1, :])
        return out


input_dim = 28

# At each time step we fit only 28 pixels and hence we have 28 time steps to equal 28* 28 pixels
layer_dim = 1 # CHANGE HERE FOR 1 LAYER to 2 LAYERS
hidden_dim = 100
out_dim = 10
model = RecurrentNeuralNetwork(input_dim, hidden_dim, layer_dim, out_dim)
criterion = nn.CrossEntropyLoss()

learning_rate = 0.1

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Parameter sizes
# Length of number of parameter sets
print(len(list(model.parameters())))
# We have 6 different groups of parameters as follows:
# 2 for Input to Hidden layer
# 2 for Hidden Layer to Output
# and 2 for Hidden layer to hidden layer for each consecutive time step


#NUMBER OF PARAMETERS CHANGE WHEN LAYERS INCREASE
# 10 SETS OF PARAMETRS FOR 2 HIDDEN LAYERS
# Input -> Hidden (A1) -> (100, 28)
print(list(model.parameters())[0].size())
# Input -> Hidden BIAS (B1) -> (100)
print(list(model.parameters())[1].size())
# Hidden -> Hidden (A3) -> (100, 100)
print(list(model.parameters())[2].size())
# Hidden -> Hidden BIAS (B3) -> (100)
print(list(model.parameters())[3].size())
# Hidden -> Output (A2) -> (10, 100)
print(list(model.parameters())[4].size())
# Hidden -> Output BIAS (B2) -> (10)
print(list(model.parameters())[5].size())

iter = 0
# Number of steps to unroll
seq_dim = 28
for i in range(num_epochs):
    for (images, labels) in train_loader:
        images = images.view(-1, seq_dim, input_dim)
        optimizer.zero_grad()
        outputs = model(images)
        # if torch.isnan(outputs).sum():
        #     print("iterartion is {}", iter)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        iter += 1
        if iter % 500 == 0:
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.view(-1, seq_dim, input_dim)
                outputs = model(images)
                predicted = torch.max(outputs.data, 1)[1]
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            accuracy = (correct / total) * 100

            print("Iteration: {}, Loss: {}, Accuracy: {}".format(iter, loss.item(), accuracy))
