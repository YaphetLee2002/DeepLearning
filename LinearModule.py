import torch
import torch.nn as nn
import torch.optim.lr_scheduler
from sklearn.datasets import load_boston


class LinearModule(nn.Module):
    def __init__(self, ndim):
        super(LinearModule, self).__init__()

        self.ndim = ndim
        self.weight = nn.Parameter(torch.randn(ndim, 1))
        self.bias = nn.Parameter(torch.randn(1))

    def forward(self, x):
        return x.mm(self.weight) + self.bias


def main():
    # lm = LinearModule(5)
    # x = torch.randn(4, 5)
    # print(lm(x))
    # print(lm.named_parameters())
    # print(list(lm.named_parameters()))

    boston = load_boston()

    lm = LinearModule(13)
    criterion = nn.MSELoss()  # 构建损失函数的计算模块
    optim = torch.optim.SGD(lm.parameters(), lr=1e-6)  # 随机梯度下降算法优化器
    data = torch.tensor(boston['data'], requires_grad=True, dtype=torch.float32)  # 传入的数据为Numpy双精度数组，需要转化为单精度
    target = torch.tensor(boston['target'], dtype=torch.float32)

    for step in range(5000):
        predict = lm(data)  # 输出模型预测结果
        loss = criterion(predict, target)  # 输出损失函数
        if step and step % 100 == 0:
            print("Loss: {:.3f}".format(loss.item()))
        optim.zero_grad()  # 清零梯度
        loss.backward()  # 反向传播
        optim.step()
        scheduler = torch.optim.lr_scheduler.StepLR

if __name__ == '__main__':
    main()
