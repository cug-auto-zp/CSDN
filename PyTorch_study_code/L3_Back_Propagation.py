import torch
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = torch.tensor([1.0], requires_grad=True)

def forward(x):
    return x * w

def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2

print('Predict (before training)',4,forward(4).item())

l_list = []
epoch_list = []
for epoch in range(100):
    for x,y in zip(x_data, y_data):
        l = loss(x,y)
        l.backward()
        print("\tgrad: ",x,y,w.grad.item())
        w.data = w.data - 0.01 * w.grad.data
        w.grad.data.zero_()

    print('progress:', epoch, l.item())
    l_list.append(l.item())
    epoch_list.append(epoch)
print('Predict (after training)',4,forward(4).item())

plt.plot(epoch_list, l_list)
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.title('Back Propagation')
plt.show()

