'''
    加载pytorch自带的模型，从头训练自己的数据
'''
import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from utils import LoadData

from torchvision.models import alexnet  #最简单的模型
from torchvision.models import vgg11, vgg13, vgg16, vgg19   # VGG系列
from torchvision.models import resnet18, resnet34,resnet50, resnet101, resnet152    # ResNet系列
from torchvision.models import inception_v3     # Inception 系列
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

# print(torch.cuda.device_count())
print(torch.cuda.is_available())

# 定义训练函数，需要
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # 从数据加载器中读取batch（一次读取多少张，即批次数），X(图片数据)，y（图片真实标签）。
    for batch, (X, y) in enumerate(dataloader):
        # 将数据存到显卡
        X, y = X.cuda(), y.cuda()

        # 得到预测的结果pred
        pred = model(X)

        # 计算预测的误差
        # print(pred,y)
        loss = loss_fn(pred, y)

        # 反向传播，更新模型参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 每训练10次，输出一次当前信息
        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model):
    size = len(dataloader.dataset)
    # 将模型转为验证模式
    model.eval()
    # 初始化test_loss 和 correct， 用来统计每次的误差
    test_loss, correct = 0, 0
    # 测试时模型参数不用更新，所以no_gard()
    # 非训练， 推理期用到
    with torch.no_grad():
        # 加载数据加载器，得到里面的X（图片数据）和y(真实标签）
        for X, y in dataloader:
            # 将数据转到GPU
            X, y = X.cuda(), y.cuda()
            # 将图片传入到模型当中就，得到预测的值pred
            pred = model(X)
            # 计算预测值pred和真实值y的差距
            test_loss += loss_fn(pred, y).item()
            # 统计预测正确的个数

            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    print(f"correct = {correct}, Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")




if __name__=='__main__':
    batch_size = 32

    # # 给训练集和测试集分别创建一个数据集加载器
    train_data = LoadData("train.txt", True)
    valid_data = LoadData("test.txt", False)


    train_dataloader = DataLoader(dataset=train_data, num_workers=4, pin_memory=True, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset=valid_data, num_workers=4, pin_memory=True, batch_size=batch_size)

    # 如果显卡可用，则用显卡进行训练
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")


    '''
        随着模型的加深，需要训练的模型参数量增加，相同的训练次数下模型训练准确率起来得更慢
    '''

    '''        ResNet系列    '''
    model = vgg13(pretrained=False, num_classes=5).to(device)  # 43.6%

    '''        ResNet系列    '''
    # model = resnet18(pretrained=False, num_classes=5).to(device)    # 43.6%

    print(model)
    # 定义损失函数，计算相差多少，交叉熵，
    loss_fn = nn.CrossEntropyLoss()

    # 定义优化器，用来训练时候优化模型参数，随机梯度下降法
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # 初始学习率


    # 一共训练1次
    epochs = 100
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        # time_start = time.time()
        # train(train_dataloader, model, loss_fn, optimizer)
        # time_end = time.time()
        # print(f"train time: {(time_end-time_start)}")
        test(test_dataloader, model)
    print("Done!")

    # 保存训练好的模型
    torch.save(model.state_dict(), "model_resnet18_100.pth")
    print("Saved PyTorch Model Success!")