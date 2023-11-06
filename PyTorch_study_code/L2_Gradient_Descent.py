import numpy as np
import matplotlib.pyplot as plt

# Prepare the training set
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

# Initial guess of weigt
w = 1.0

# Define the model
def forward(x):
    return x * w

# Define the cost function
def cost(xs, ys):
    cost = 0
    for x, y in zip(xs, ys):
        y_pred = forward(x)
        cost += (y_pred - y) ** 2
    return cost / len(xs)

# Define the gradient function
def gradient(xs, ys):
    grad = 0
    for x, y in zip(xs, ys):
        grad += 2 * x * (forward(x) - y)
    return grad / len(xs)

print('Predict (before training)', 4, forward(4))

epoch_list = []
cost_list = []
for epoch in range(100):
    grad = gradient(x_data, y_data)
    w -= 0.01 * grad
    epoch_list.append(epoch)
    cost_list.append(cost(x_data, y_data))
    print('Epoch:', epoch, 'w=', w, 'cost=', cost(x_data, y_data))

print('Predict (after training)', 4, forward(4))
plt.plot(epoch_list, cost_list)
plt.ylabel('Cost')
plt.xlabel('epoch')
plt.title('Gradient Descent')
plt.savefig('PyTorch_study_code\Fig\L2_GD.png')
plt.show()