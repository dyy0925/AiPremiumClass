import torch.nn as nn

# 定义GRU模型
class GRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(input_size=64, hidden_size=128, num_layers=2, batch_first=True)
        self.fc = nn.Linear(128, 40)

    def forward(self, x):
        y, _ = self.gru(x)
        yc = self.fc(y[:, -1, :])
        return yc