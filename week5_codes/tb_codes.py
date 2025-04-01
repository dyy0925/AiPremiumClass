from torch.utils.tensorboard import SummaryWriter
import numpy as np
from torchvision import transforms,datasets 
import torch
import torchvision

# 随机数
writer = SummaryWriter('runs/random_data')
for n_iter in range(100):
    writer.add_scalar('Loss/train', np.random.random(), n_iter)
    writer.add_scalar('Loss/test', np.random.random(), n_iter)
    writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
    writer.add_scalar('Accuracy/test', np.random.random(), n_iter)
writer.close()

# 直线
writer = SummaryWriter('runs/linear_data')
x = range(100)
for i in x:
    writer.add_scalar('y=2x', i * 2, i)
writer.close()

# 曲线
writer = SummaryWriter('runs/curve_data')
r = 5
for i in range(100):
    writer.add_scalars('run_14h', {'xsinx':i*np.sin(i/r),
                                    'xcosx':i*np.cos(i/r),
                                    'tanx': np.tan(i/r)}, i)    
writer.close()

# 立体
writer = SummaryWriter('runs/histogram_data')    
for i in range(10):
    x = np.random.random(1000)
    writer.add_histogram('distribution centers', x + i, i)
    
writer.close()

# 计算图-不好理解-again
writer = SummaryWriter('runs/image_data')

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = datasets.MNIST('mnist_train', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
images, labels = next(iter(trainloader))
grid = torchvision.utils.make_grid(images)
writer.add_image('images', grid, 0)

model = torchvision.models.resnet50(False)
# Have ResNet model take in grayscale rather than RGB
model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
writer.add_graph(model, images)

writer.close()