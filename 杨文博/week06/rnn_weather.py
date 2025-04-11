import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

file_path = r'D:\work\code\practice\home_work\杨文博\week06\weather_data\Summary of Weather.csv'
df = pd.read_csv(file_path)

print(df.head())
data = df['MaxTemp'].values.astype(float)
scaler = MinMaxScaler(feature_range=(0, 1))
data_reshape = data.reshape(-1, 1)

def create_dataset(data, look_back=5):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back), 0])
        y.append(data[i + look_back, 0])
    return np.array(X), np.array(y)

look_back = 10
X, y = create_dataset(data_reshape, look_back)
X = torch.FloatTensor(X).reshape(-1,look_back,1)
y = torch.FloatTensor(y).reshape(-1, 1)

print(X.shape, y.shape)

class SimpleRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, output_size=1):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        rnn_out, _ = self.rnn(x)
        # rnn_out = self.dropout(rnn_out)  # 应用 Dropout
        output = self.fc(rnn_out[:, -1, :])
        return output

train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

batch_size = 32
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
device = torch.device('cuda')
model = SimpleRNN().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}')
    writer.add_scalar('Loss/train', loss, epoch)

for batch_X, batch_y in test_loader:
    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
    outputs = model(batch_X)
    loss = criterion(outputs, batch_y)
    print(f'Test Loss: {loss.item():.4f}')

