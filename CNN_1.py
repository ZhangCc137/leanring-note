import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

def get_num_correct(preds,labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5)
        self.conv2=nn.Conv2d(in_channels=6,out_channels=12,kernel_size=5)
        
        self.fc1=nn.Linear(in_features=12*4*4,out_features=120)
        self.fc2=nn.Linear(in_features=120,out_features=60)
        self.out=nn.Linear(in_features=60,out_features=10)
        
    def forward(self,t):
        
        # (1) input layer
        t=t
        
        # (2) hidden conv layer
        t=self.conv1(t)
        t=F.relu(t)
        t=F.max_pool2d(t,kernel_size=2,stride=2)
        
        # (3) hidden conv layer
        t=self.conv2(t)
        t=F.relu(t)
        t=F.max_pool2d(t,kernel_size=2,stride=2)
        
        # (4) hidden liner layer
        t=t.reshape(-1,12*4*4)  # 带batch的二阶张量
        t=self.fc1(t)
        t=F.relu(t)
        
        # (5) hidden liner layer
        t=self.fc2(t)
        t=F.relu(t)
        
        # (6) output layer 
        t=self.out(t)
        # t=F.softmax(t,dim=1)
        
        return t

network=Network()

# 获取训练集
train_set=torchvision.datasets.FashionMNIST(
    root='./data'
    ,train=True
    ,download=True
    ,transform=transforms.Compose([
        transforms.ToTensor()
    ])
)

# 传入批数据
train_loader=torch.utils.data.DataLoader(train_set,batch_size=100)
optimizer=optim.Adam(network.parameters(),lr=0.01)



for epoch in range(10):

    total_loss=0
    total_correct=0
    
    for batch in train_loader:
        images,labels=batch

        # 计算损失函数
        preds=network(images)
        loss=F.cross_entropy(preds,labels)

        # 计算梯度,更新权重
        optimizer.zero_grad() # 梯度归零
        loss.backward()
        optimizer.step()

        total_loss+=loss.item()
        total_correct+=get_num_correct(preds,labels)

    print("epoch:",epoch,"total_loss:",total_loss,"total_correct:",total_correct)

print(total_correct/len(train_set))
