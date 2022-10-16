import warnings
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from sklearn import preprocessing
# from IPython.display import display
import torch
import torch.optim as optim

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    # 获取数据
    features = pd.read_csv('temps.csv')
    features.head()
    # 获取年月日
    years = features['year']
    months = features['month']
    days = features['day']
    # 获取datetime格式数据
    dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in
             zip(years, months, days)]
    dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]

    # 准备画图
    # 指定默认风格
    plt.style.use('fivethirtyeight')
    # 设置布局
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))
    fig.autofmt_xdate(rotation=45)
    # 标签值
    ax1.plot(dates, features['actual'])
    ax1.set_xlabel('')
    ax1.set_ylabel('Temperature')
    ax1.set_title('Max Temp')

    # 昨天
    ax2.plot(dates, features['temp_1'])
    ax2.set_xlabel('')
    ax2.set_ylabel('Temperature')
    ax2.set_title('Previous Max Temp')

    # 前天
    ax3.plot(dates, features['temp_2'])
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Temperature')
    ax3.set_title('Two Days Prior Max Temp')

    # 我的逗逼朋友
    ax4.plot(dates, features['friend'])
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Temperature')
    ax4.set_title('Friend Estimate')

    plt.tight_layout(pad=2)
    plt.show()
    # 独热编码
    features = pd.get_dummies(features)
    # display(features)

    # 获取标签值
    labels = np.array(features['actual'])
    # 在 特征中去掉标签
    features = features.drop('actual', axis=1)
    # 保存列名
    features_list = list(features.columns)
    # 转换为numpy的ndArray
    features = np.array(features)

    # StandardScaler：去均值和方差归一化
    # fit_transform：序列重排后标准化
    input_features = preprocessing.StandardScaler().fit_transform(features)

    # # 构建网络模型
    # x = torch.tensor(input_features, dtype=float)
    # y = torch.tensor(labels, dtype=float)
    #
    # # 权重参数初始化
    # weights = torch.randn((14, 128), dtype=torch.float, requires_grad=True)
    # biases = torch.randn(128, dtype=torch.float, requires_grad=True)
    # weights2 = torch.randn((128, 1), dtype=torch.float, requires_grad=True)
    # biases2 = torch.randn(1, dtype=torch.float, requires_grad=True)
    #
    # learning_rate = 0.001
    # losses = []

    # for i in range(1000):
    #     # 计算隐层
    #     hidden = x.mm(weights) + biases
    #     # 加入激活函数
    #     hidden = torch.relu(hidden)
    #     # 预测结果
    #     predictions = hidden.mm(weights2) + biases2
    #     # 计算损失
    #     loss = torch.mean((predictions - y) ** 2)
    #     losses.append(loss.data.numpy())
    #     # 打印损失值
    #     if i % 100 == 0:
    #         print('loss:', i, loss)
    #     # 反向传播计算
    #     loss.backward()
    #     # 更新参数
    #     weights.data.add_(-learning_rate * weights.grad.data)
    #     biases.data.add_(-learning_rate * biases.grad.data)
    #     weights2.data.add_(-learning_rate * weights2.grad.data)
    #     biases2.data.add_(-learning_rate * biases2.grad.data)
    #     # 每次清空grad.data
    #     weights.grad.data.zero_()
    #     biases.grad.data.zero_()
    #     weights2.grad.data.zero_()
    #     biases2.grad.data.zero_()

    # 更简单的构建网络模型
    input_size = input_features.shape[1]
    hidden_size = 128
    output_size = 1
    batch_size = 16
    my_nn = torch.nn.Sequential(
        torch.nn.Linear(input_size, hidden_size),
        torch.nn.Sigmoid(),
        torch.nn.Linear(hidden_size, output_size)
    )
    cost = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(my_nn.parameters(), lr=0.001)

    # 训练网络
    losses = []
    for i in range(1000):
        batch_loss = []
        # MINI-Batch方法来进行训练
        for start in range(0, len(input_features), batch_size):
            end = start + batch_size if start + batch_size < len(input_features) else len(input_features)
            xx = torch.tensor(input_features[start:end], dtype=torch.float, requires_grad=True)
            yy = torch.tensor(labels[start:end], dtype=torch.float, requires_grad=True)
            prediction = my_nn(xx)
            loss = cost(prediction, yy)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            batch_loss.append(loss.data.numpy())

        # 打印损失
        if i % 100 == 0:
            losses.append(np.mean(batch_loss))
            print(i, np.mean(batch_loss))

    x = torch.tensor(input_features, dtype=torch.float)
    predict = my_nn(x).data.numpy()

    # 转换日期格式
    dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in
             zip(years, months, days)]
    dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]
    # 存储日期和对应标签值
    true_data = pd.DataFrame(data={'date': dates, 'actual': labels})
    # 存储日期和对应的模型预测值
    months = features[:, features_list.index('month')]
    days = features[:, features_list.index('day')]
    years = features[:, features_list.index('year')]

    test_dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in
                  zip(years, months, days)]
    test_dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in test_dates]

    predictions_data = pd.DataFrame(data={'date': test_dates, 'prediction': predict.reshape(-1)})

    # 绘制标签值曲线图
    plt.plot(true_data['date'], true_data['actual'], 'b-', label='actual')
    # 绘制预测值曲线图
    plt.plot(predictions_data['date'], predictions_data['prediction'], 'y-', label='prediction')
    # 设置横坐标步长
    plt.xticks(rotation='60')
    # 添加图例
    plt.legend()
    # 设置表名
    plt.xlabel('Date')
    plt.ylabel('Maximum Temperature(F°)')
    plt.title('Actual and predict temperature')
    plt.tight_layout()
    plt.show()
