import torch
import matplotlib.pyplot as plt

# Generate some data
torch.manual_seed(42)
x = torch.linspace(-1, 1, 100)
y = 2*x + 3 + 0.3*torch.randn(x.size())

# Initialize weight and bias
w = torch.randn(1, requires_grad=False)
b = torch.randn(1, requires_grad=False)

# Learning rate
lr = 0.1

# Number of epochs
epochs = 100

losses = []

for epoch in range(epochs):
    # Compute model's predictions
    y_pred = w * x + b

    # Compute loss
    loss = ((y - y_pred) ** 2).mean()
    losses.append(loss.item())

    # Compute gradients
    w_grad = -2 * (x * (y - y_pred)).mean()
    b_grad = -2 * (y - y_pred).mean()

    # Update weights using gradient descent
    w -= lr * w_grad
    b -= lr * b_grad

    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# Plot the loss over time
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(range(epochs), losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss over time')

# Plot the original data and learned line
plt.subplot(1, 2, 2)
plt.scatter(x, y, color='blue')
plt.plot(x, w.item()*x+b.item(), color='red', linewidth=2)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Data and Trained Line')

plt.tight_layout()
plt.show()
