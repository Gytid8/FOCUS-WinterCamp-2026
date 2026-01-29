import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # 强制使用 Tk 渲染引擎
import matplotlib.pyplot as plt
# 制作数据集
# 数据集转换参数
transform = transforms.Compose([
 transforms.ToTensor(),
 transforms.Normalize(0.1307, 0.3081)
])
# 下载训练集与测试集
train_Data = datasets.MNIST(
 root = 'D:/Jupyter/dataset/mnist/', # 下载路径
 train = True, # 是 train 集
 download = True, # 如果该路径没有该数据集，就下载
 transform = transform # 数据集转换参数
)
test_Data = datasets.MNIST(
 root = 'D:/Jupyter/dataset/mnist/', # 下载路径
 train = False, # 是 test 集
 download = True, # 如果该路径没有该数据集，就下载
 transform = transform # 数据集转换参数
)
 # 批次加载器
train_loader = DataLoader(train_Data, shuffle=True, batch_size=256)
test_loader = DataLoader(test_Data, shuffle=False, batch_size=256)


class CNN(nn.Module):
    def __init__(self):
        ''' 搭建神经网络各层 '''
        super(CNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Tanh(), # C1：卷积层
            nn.AvgPool2d(kernel_size=2, stride=2), # S2：平均汇聚
            nn.Conv2d(6, 16, kernel_size=5), nn.Tanh(), # C3：卷积层
            nn.AvgPool2d(kernel_size=2, stride=2), # S4：平均汇聚
            nn.Conv2d(16, 120, kernel_size=5), nn.Tanh(), # C5：卷积层
            nn.Flatten(), # 把图像铺平成一维
            nn.Linear(120, 84), nn.Tanh(), # F5：全连接层
            nn.Linear(84, 10) # F6：全连接层
        )
    def forward(self, x):
        y = self.net(x)
        return y

# 查看网络结构
X = torch.rand(size=(1, 1, 28, 28))
for layer in CNN().net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape: \t', X.shape)
# 创建子类的实例，并搬到 GPU 上
model = CNN().to('cuda:0')
# 损失函数的选择
loss_fn = nn.CrossEntropyLoss() # 自带 softmax 激活函数
# 优化算法的选择
learning_rate = 0.9 # 设置学习率
optimizer = torch.optim.SGD(
    model.parameters(),
    lr = learning_rate,
)
epochs = 5
losses = []
for epoch in range(epochs):
    for (x, y) in train_loader:
        x, y = x.to('cuda:0'), y.to('cuda:0')
        Pred = model(x)
        loss = loss_fn(Pred, y)
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
Fig = plt.figure()
plt.plot(range(len(losses)), losses)
plt.show()
correct = 0
total = 0
with torch.no_grad():
    for (x, y) in test_loader:
        x, y = x.to('cuda:0'), y.to('cuda:0')
        Pred = model(x)
        _, predicted = torch.max(Pred.data, dim=1)
        correct += torch.sum( (predicted == y) )
        total += y.size(0)
print(f'测试集精准度: {100*correct/total} %')