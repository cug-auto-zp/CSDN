# import torch
# in_channels, out_channels= 5, 10
# width, height = 100, 100
# kernel_size = 3
# batch_size = 1
# input = torch.randn(batch_size,
# in_channels,
# width, 
# height)
# conv_layer = torch.nn.Conv2d(in_channels,
# out_channels,
# kernel_size=kernel_size)
# output = conv_layer(input)
# print(input.shape)
# print(output.shape)
# print(conv_layer.weight.shape)


# import torch
# input = [3,4,6,5,7,
#          2,4,6,8,2,
#          1,6,7,8,4,
#          9,7,4,6,2,
#          3,7,5,4,1]
# input = torch.Tensor(input).view(1, 1, 5, 5)
# conv_layer = torch.nn.Conv2d(1, 1, kernel_size=3, stride = 2, bias=False)
# kernel = torch.Tensor([1,2,3,4,5,6,7,8,9]).view(1, 1, 3, 3)
# conv_layer.weight.data = kernel.data
# output = conv_layer(input)
# print(output)

import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

# 1. Prepare Dataset

batch_size = 64
transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.1307, ), (0.3081, ))
    ])
train_dataset = datasets.MNIST(root='PyTorch_study_code/Dataset/mnist/',
                                train=True,
                                download=True,
                                transform=transform)
train_loader = DataLoader(train_dataset,
                            shuffle=True,   
                            batch_size=batch_size)
test_dataset = datasets.MNIST(root='PyTorch_study_code/dataset/mnist/',
                                train=False,
                                download=True,
                                transform=transform)
test_loader = DataLoader(test_dataset,
                            shuffle=False,
                            batch_size=batch_size)

# 2. Design Model

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.pooling = torch.nn.MaxPool2d(2)
        self.fc = torch.nn.Linear(320, 10)
    def forward(self, x):
    # Flatten data from (n, 1, 28, 28) to (n, 784)
        batch_size = x.size(0)
        x = F.relu(self.pooling(self.conv1(x)))
        x = F.relu(self.pooling(self.conv2(x)))
        x = x.view(batch_size, -1) # flatten
        x = self.fc(x)
        return x
model = Net()

# Define device as the first visible cuda device if we have CUDA available.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 3. Construct Loss and Optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


# 4. Train and Test
def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        inputs, target = inputs.to(device), target.to(device)
        optimizer.zero_grad()
        # forward + backward + update
        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0.0

def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, target = data
            inputs, target = inputs.to(device), target.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, dim=1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        print('Accuracy on test set: %d %%  [%d/%d]' % (100 * correct / total, correct, total))
    return correct / total
if __name__ == '__main__':
    epoch_list = []
    acc_list = []
    for epoch in range(10):
        train(epoch)
        acc = test()
        acc_list.append(acc)
        epoch_list.append(epoch)

    plt.plot(epoch_list, acc_list)
    plt.ylabel('Accuracy')
    plt.xlabel('epoch')
    plt.savefig('PyTorch_study_code\Fig\L8_Basic_CNN.png')
    plt.show() 






