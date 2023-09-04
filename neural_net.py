# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
torch.manual_seed(42)

# 1. Generating some random data
# Assume we have a linear relationship: y = 2x + 3 + noise
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
y = 2*x + 3 + 0.3*torch.randn(x.size())

# 2. Building the neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(1, 1)  # Single-layer perceptron

    def forward(self, x):
        return self.fc(x)

net = Net()

# 3. Defining the loss function and optimizer
loss_func = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.2)

# 4. Training the network
epochs = 200
losses = []

for epoch in range(epochs):
    # Forward pass: Compute predicted y by passing x to the model
    prediction = net(x)

    # Compute the loss
    loss = loss_func(prediction, y)
    losses.append(loss.item())

    # Zero gradients, perform backward pass, and update weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# Plot the loss over time
plt.plot(range(epochs), losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss over time')
plt.show()

# Test the trained network
test_x = torch.unsqueeze(torch.linspace(-2, 2, 100), dim=1)
test_y = net(test_x).detach()
plt.scatter(x, y, color='blue')
plt.plot(test_x, test_y, color='red', linewidth=2)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Data and Trained Line')
plt.show()
