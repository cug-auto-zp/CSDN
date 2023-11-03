import numpy as np
import matplotlib.pyplot as plt
import torch

x_data = [1.0,2.0,3.0]
y_data = [2.0,4.0,6.0]

w1 = torch.tensor([1.0], requires_grad = True)
w2 = torch.tensor([1.0], requires_grad = True)
b = torch.tensor([1.0], requires_grad = True)


def forward(x):
    return w1 * x**2 + w2 * x + b

def loss(x,y):
    y_pred = forward(x)
    return (y_pred-y) **2

print('Predict (befortraining)',4,forward(4))

l_list = []
epoch_list = []
for epoch in range(100):
    for x,y in zip(x_data,y_data):
        l = loss(x, y)
        l.backward()
        print('\tgrad:',x,y,w1.grad.item(),w2.grad.item(),b.grad.item())
        w1.data = w1.data - 0.02*w1.grad.data 
        w2.data = w2.data - 0.02 * w2.grad.data
        b.data = b.data - 0.02 * b.grad.data
        w1.grad.data.zero_() 
        w2.grad.data.zero_()
        b.grad.data.zero_()
    print('Epoch:',epoch,l.item())
    l_list.append(l.item())
    epoch_list.append(epoch)

print('Predict(after training)',4,forward(4).item())
plt.plot(epoch_list, l_list)
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.title('Back Propagation')
plt.show()