import torch
import matplotlib.pyplot as plt
import numpy as np
x_data = torch.tensor([[1.0], [2.0], [3.0]])
y_data = torch.tensor([[2.0], [4.0], [6.0]])    


class LinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

model = LinearModel()
criterion = torch.nn.MSELoss(reduction='sum')                   # LBFGS 优化器时候需要使用这句话
optimizer = torch.optim.LBFGS(model.parameters(), lr=0.01)

epoch_list = []         # 保存epoch迭代
loss_list = []          # 保存每次迭代的loss值

"""
    LBFGS 优化器时会出现这个报错: step() missing 1 required positional argument: 'closure', 为了解决此问题找到了以下两个链接解决此类问题
    解决问题链接: https://blog.csdn.net/bit452/article/details/109677086
               : https://blog.csdn.net/xian0710830114/article/details/128419401
"""

for epoch in range(100):
    def closure():
        y_pred = model(x_data)
        loss = criterion(y_pred, y_data)
        optimizer.zero_grad()
        loss.backward()
        return loss
    loss = optimizer.step(closure=closure)
    print(epoch, loss.item())
    loss_list.append(loss.item())
    epoch_list.append(epoch)
print('w = ',model.linear.weight.item())
print('b = ',model.linear.bias.item())

x_test = torch.tensor([[4.0]])
y_test = model(x_test)
print('y_pred = ',y_test.data)

plt.plot(epoch_list, loss_list)
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.title('LBFGS')
plt.savefig('PyTorch_study_code\Fig\L4_LBFGS.png')
plt.show()




