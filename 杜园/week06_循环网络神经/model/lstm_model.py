import torch.nn as nn

# 定义LSTM模型
class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=64, hidden_size=128, num_layers=2, batch_first=True)
        self.fc = nn.Linear(128, 40)

    def forward(self, x):
        y, _ = self.lstm(x)
        yc = self.fc(y[:, -1, :])
        return yc