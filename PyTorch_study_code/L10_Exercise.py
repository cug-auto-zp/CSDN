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

# Residual Block
class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.channels = channels
        self.conv1 = torch.nn.Conv2d(channels, channels, kernel_size=3,padding=1)
        self.conv2 = torch.nn.Conv2d(channels, channels, kernel_size=3,padding=1)

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = self.conv2(y)
        return F.relu(x + y)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=5)
        self.mp = torch.nn.MaxPool2d(2)
        self.rblock1 = ResidualBlock(16)
        self.rblock2 = ResidualBlock(32)
        self.fc = torch.nn.Linear(512, 10)
    def forward(self, x):
        in_size = x.size(0)
        x = self.mp(F.relu(self.conv1(x)))
        x = self.rblock1(x)
        x = self.mp(F.relu(self.conv2(x)))
        x = self.rblock2(x)
        x = x.view(in_size, -1)
        x = self.fc(x)
        return x
model = Net()

# # 文献[1] He K, Zhang X, Ren S, et al. Identity Mappings in Deep Residual Networks[C]
# # (b) constant scaling
# class ResidualBlockb(torch.nn.Module):
#     def __init__(self, channels):
#         super(ResidualBlockb, self).__init__()
#         self.channels = channels
#         self.conv1 = torch.nn.Conv2d(channels, channels, kernel_size=3,padding=1)
#         self.conv2 = torch.nn.Conv2d(channels, channels, kernel_size=3,padding=1)

#     def forward(self, x):
#         y = F.relu(self.conv1(x))
#         y = self.conv2(y)
#         return F.relu(0.5*(x + y))

# class Netb(torch.nn.Module):
#     def __init__(self):
#         super(Netb, self).__init__()
#         self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=5)
#         self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=5)
#         self.mp = torch.nn.MaxPool2d(2)
#         self.rblockb1 = ResidualBlockb(16)
#         self.rblockb2 = ResidualBlockb(32)
#         self.fc = torch.nn.Linear(512, 10)
#     def forward(self, x):
#         in_size = x.size(0)
#         x = self.mp(F.relu(self.conv1(x)))
#         x = self.rblockb1(x)
#         x = self.mp(F.relu(self.conv2(x)))
#         x = self.rblockb2(x)
#         x = x.view(in_size, -1)
#         x = self.fc(x)
#         return x

# model = Netb()

# # (e) conv shortcut
# class ResidualBlocke(torch.nn.Module):
    # def __init__(self, channels):
        # super(ResidualBlocke, self).__init__()
        # self.channels = channels
        # self.conv1 = torch.nn.Conv2d(channels, channels, kernel_size=3,padding=1)
        # self.conv2 = torch.nn.Conv2d(channels, channels, kernel_size=3,padding=1)
# 
        # self.conv3 = torch.nn.Conv2d(channels, channels, kernel_size=1)
# 
    # def forward(self, x):
        # x = self.conv3(x)
        # y = F.relu(self.conv1(x))
        # y = self.conv2(y)
        # return F.relu(x + y)
# 
# class Nete(torch.nn.Module):
    # def __init__(self):
        # super(Nete, self).__init__()
        # self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=5)
        # self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=5)
        # self.mp = torch.nn.MaxPool2d(2)
        # self.rblockb1 = ResidualBlocke(16)
        # self.rblockb2 = ResidualBlocke(32)
        # self.fc = torch.nn.Linear(512, 10)
    # def forward(self, x):
        # in_size = x.size(0)
        # x = self.mp(F.relu(self.conv1(x)))
        # x = self.rblockb1(x)
        # x = self.mp(F.relu(self.conv2(x)))
        # x = self.rblockb2(x)
        # x = x.view(in_size, -1)
        # x = self.fc(x)
        # return x
# 
# model = Nete()


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
        epoch_list.append(epoch + 1)

    plt.plot(epoch_list, acc_list)
    plt.ylabel('Accuracy')
    plt.xlabel('epoch')
    plt.title('(a) original')
    plt.savefig('PyTorch_study_code\Fig\L10_(a)original.png')
    plt.show()