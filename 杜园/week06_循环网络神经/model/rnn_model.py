import torch.nn as nn
import torch

class RNN(nn.Module):
    
    def __init__(self):
        super().__init__()
        # batch_first=False :(每个样本中时间步的数量, 每个批次中的样本数量, 每个时间步的特征数量)
        # batch_first=True :(每个批次中的样本数量, 每个样本中时间步的数量, 每个时间步的特征数量)
        self.Rnn = nn.RNN(input_size=64, hidden_size=128, num_layers=2, batch_first=True)
        self.fc = nn.Linear(128, 40)
        
    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), 128).to(x.device)
        # x(32, 64, 64) 
        y, _ = self.Rnn(x, h0)
        # y(32, 64, 128)
        out = y[:, -1, :]
        # out(32, 128) 
        # 提取出每个样本在最后一个时间步的特征向量
        yc = self.fc(out)
        return yc