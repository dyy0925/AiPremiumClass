import torch.nn as nn

class RNNModel(nn.Module):
    def __init__(self, seq_length):
        super().__init__()
        # （输入层x的特征数量，隐藏状态特征数，...）
        self.rnn = nn.RNN(1, 64, 1, batch_first=True)
        self.fc_1day = nn.Linear(64, 1)
        self.fc_5days = nn.Linear(64, 1 * seq_length)
        self.seq_length = seq_length

    def forward(self, x):
        out, _ = self.rnn(x)
        last_hidden_state = out[:, -1, :]  # 取最后一个时间步的隐藏状态
        y_1day = self.fc_1day(last_hidden_state)
        y_5days = self.fc_5days(last_hidden_state).view(-1, self.seq_length, 1)
        return y_1day, y_5days