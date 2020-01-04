import torch
import numpy as np
import torch.nn as nn

x_values = list(range(11))
x_train = np.array(x_values, dtype=np.float32)
print(x_train.shape)

x_train = x_train.reshape(-1, 1)
print(x_train.shape)

y_values = [2 * i + 1 for i in x_values]
print(y_values)

y_train = np.array(y_values, dtype=np.float32)
y_train = y_train.reshape(-1, 1)
print(y_train.shape)


class LinearRegressionModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        out = self.linear(x)
        return out


input_dim = 1
out_dim = 1

model = LinearRegressionModel(input_dim, out_dim)
criterion = nn.MSELoss()
if torch.cuda.is_available():
    model.cuda()

learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

epochs = 10

for epoch in range(epochs):
    inputs = torch.from_numpy(x_train)
    labels = torch.from_numpy(y_train)
    # Clear gradients
    if torch.cuda.is_available():
        inputs.cuda()
        labels.cuda()
    optimizer.zero_grad()
    outputs = model(inputs)

    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    print('epoch {}, loss {}'.format(epoch + 1, loss.item()))

# torch.save(model.state_dict(), "liner.pk1")

model.load_state_dict(torch.load('liner.pk1'))
print(model(inputs).data.numpy())
print(model.parameters())
