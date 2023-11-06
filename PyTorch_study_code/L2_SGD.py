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

# Define the loss function
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2

# Define the gradient function
def gradient(x, y):
    return 2 * x * (forward(x) - y)   

print('Predict (before training)', 4, forward(4))

epoch_list = []
cost_list = []
for epoch in range(100):
    for x, y in zip(x_data, y_data):
        grad = gradient(x, y)

        # update weight by every grad of sample of training set
        w -= 0.01 * grad
        print('\tgrad:',x,y,grad)
    epoch_list.append(epoch)
    cost_list.append(loss(x, y))
    print('Epoch:', epoch, 'w=', w, 'cost=', loss(x, y))

print('Predict (after training)', 4, forward(4))
plt.plot(epoch_list, cost_list)
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.title('Stochastic Gradient Descent')
plt.savefig('PyTorch_study_code\Fig\L2_Stochastic_Gradient_Descent.png')
plt.show()