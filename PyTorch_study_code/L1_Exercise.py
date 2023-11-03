import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]


def forward(x):
    return x * w + b

def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2

W = np.arange(0.0,4.1,0.1)
B = np.arange(0.0,4.1,0.1)

[w,b] = np.meshgrid(W,B)

w_list = []
mse_list = []

l_sum = 0
for x_val, y_val in zip(x_data, y_data):
    y_pred = forward(x_val)
    print(y_pred)
    l = loss(x_val, y_val)
    l_sum += l

fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(w, b, l_sum / 3, cmap='rainbow')
ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

ax.set_xlabel('w')
ax.set_ylabel('b')
ax.set_zlabel('Loss')
ax.grid(False)

plt.show()

