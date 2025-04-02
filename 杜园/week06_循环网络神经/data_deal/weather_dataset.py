import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


# 2、数据预处理
class WeatherDataset(Dataset):
    def __init__(self, data, seq_length=5):
        self.seq_length = seq_length
        self.data = data.values
        self.scaler = StandardScaler()
        self.data = self.scaler.fit_transform(self.data.reshape(-1, 1))
        self.X, self.y_1day, self.y_5days = [], [], []
        
        # 假设：len(self.data(1-10)) = 10 seq_length=3
        # 第一轮：X: 1/2/3 y_1day: 3 y_5days: 2/3/4
        for i in range(len(self.data) - seq_length):
            # 输入序列的集合(指定步长的数据为一组 )
            self.X.append(self.data[i:i + seq_length])
            # 单步预测的目标值(预测接下来一个时间步的值)
            self.y_1day.append(self.data[i + seq_length])
            # 多步预测的目标值(预测接下来多个时间步的值)
            self.y_5days.append(self.data[i + 1:i + seq_length + 1])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), \
               torch.tensor(self.y_1day[idx], dtype=torch.float32), \
               torch.tensor(self.y_5days[idx], dtype=torch.float32)