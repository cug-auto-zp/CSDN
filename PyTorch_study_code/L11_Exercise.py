#---------------------- Using LSTM ----------------------------#
# import torch

# # parameters
# num_class = 4
# input_size = 4
# hidden_size = 20
# embedding_size = 10
# num_layers = 2
# batch_size = 1
# seq_len = 5

# idx2char = ['e', 'h', 'l', 'o']
# x_data = [[1, 0, 2, 2, 3]] # (batch, seq_len)
# y_data = [3, 1, 2, 3, 2] # (batch * seq_len)
# inputs = torch.LongTensor(x_data)
# labels = torch.LongTensor(y_data)

# class Model(torch.nn.Module):
#     def __init__(self):
#         super(Model, self).__init__()
#         self.emb = torch.nn.Embedding(input_size, embedding_size)
#         self.lstm = torch.nn.LSTM(input_size=embedding_size,
#                                 hidden_size=hidden_size,
#                                 num_layers=num_layers,
#                                 batch_first=True)
#         self.fc = torch.nn.Linear(hidden_size, num_class)
#     def forward(self, x):
#         hidden = torch.zeros(num_layers, x.size(0), hidden_size)
#         c0 = torch.zeros(num_layers, x.size(0), hidden_size)
#         x = self.emb(x) # (batch, seqLen, embeddingSize)
#         x, _ = self.lstm(x, (hidden, c0))
#         x = self.fc(x)
#         return x.view(-1, num_class)

# net = Model()
# criterion = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(net.parameters(), lr=0.05)

# for epoch in range(15):
#     optimizer.zero_grad()
#     outputs = net(inputs)
#     loss = criterion(outputs, labels)
#     loss.backward()
#     optimizer.step()
#     _, idx = outputs.max(dim=1)
#     idx = idx.data.numpy()
#     print('Predicted: ', ''.join([idx2char[x] for x in idx]), end='')
#     print(', Epoch [%d/15] loss = %.3f' % (epoch + 1, loss.item()))


#---------------------- Using GRU ----------------------------#
import torch

# parameters
num_class = 4
input_size = 4
hidden_size = 20
embedding_size = 10
num_layers = 2
batch_size = 1
seq_len = 5

idx2char = ['e', 'h', 'l', 'o']
x_data = [[1, 0, 2, 2, 3]] # (batch, seq_len)
y_data = [3, 1, 2, 3, 2] # (batch * seq_len)
inputs = torch.LongTensor(x_data)
labels = torch.LongTensor(y_data)

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.emb = torch.nn.Embedding(input_size, embedding_size)
        self.gru = torch.nn.GRU(input_size=embedding_size,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, num_class)
    def forward(self, x):
        hidden = torch.zeros(num_layers, x.size(0), hidden_size)
        x = self.emb(x) # (batch, seqLen, embeddingSize)
        x, _ = self.gru(x, hidden)
        x = self.fc(x)
        return x.view(-1, num_class)

net = Model()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.05)
print('-' * 30, 'GRU', '-' * 30)
for epoch in range(15):
    optimizer.zero_grad()
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    _, idx = outputs.max(dim=1)
    idx = idx.data.numpy()
    print('Predicted: ', ''.join([idx2char[x] for x in idx]), end='')
    print(', Epoch [%d/15] loss = %.3f' % (epoch + 1, loss.item()))