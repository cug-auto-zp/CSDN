# # Cross Entropy in Numpy
# import numpy as np
# y = np.array([1, 0, 0])
# z = np.array([0.2, 0.1, -0.1])
# y_pred = np.exp(z) / np.exp(z).sum()
# loss = (- y * np.log(y_pred)).sum()
# print(loss)

# # Cross Entropy in PyTorch
# import torch
# y = torch.LongTensor([0])
# z = torch.Tensor([[0.2, 0.1, -0.1]])
# criterion = torch.nn.CrossEntropyLoss()
# loss = criterion(z, y)
# print(loss)

# # Mini-Batch: batch_size=3
# import torch
# criterion = torch.nn.CrossEntropyLoss()
# Y = torch.LongTensor([2, 0, 1])
# Y_pred1 = torch.Tensor([[0.1, 0.2, 0.9],
#                         [1.1, 0.1, 0.2],
#                         [0.2, 2.1, 0.1]])
# Y_pred2 = torch.Tensor([[0.8, 0.2, 0.3],
#                         [0.2, 0.3, 0.5],
#                         [0.2, 0.2, 0.5]])
# l1 = criterion(Y_pred1, Y)
# l2 = criterion(Y_pred2, Y)
# print("Batch Loss1 = ", l1.data, "\nBatch Loss2=", l2.data)


# # Exercise 9-1: CrossEntropyLoss vs NLLLoss
# Try to know CrossEntropyLoss <==> LogSoftmax + NLLLoss
# 此代码以 up主 ppt中Softmax Layer - Example的例子做的一些实验
import torch
y = torch.LongTensor([0])
z = torch.Tensor([[0.2, 0.1, -0.1]])
Softmax = torch.nn.Softmax(dim=1)
probs = Softmax(z)
print("Softmax: ", probs)
LogSoftmax = torch.nn.LogSoftmax(dim=1)
log_probs = LogSoftmax(z)
print("LogSoftmax: ", log_probs)
criterion = torch.nn.NLLLoss()
loss = criterion(log_probs, y)           # LogSoftmax + NLLLoss
criterion1 = torch.nn.CrossEntropyLoss() # CrossEntropyLoss
loss1 = criterion1(z, y)
print("LogSoftmax + NLLLoss: ", loss, "\nCrossEntropyLoss: ", loss1)         # 验证 CrossEntropyLoss <==> LogSoftmax + NLLLoss