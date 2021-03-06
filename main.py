
from sklearn.model_selection import train_test_split
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import torch.optim as optim
import torch
from utils.dataloader import dataload
from torchvision import transforms
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torch.autograd import Variable

# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    







    class LSTM_Regression(nn.Module):
        """
            使用LSTM进行回归

            参数：
            - input_size: feature size
            - hidden_size: number of hidden units
            - output_size: number of output
            - num_layers: layers of LSTM to stack
        """

        def __init__(self, input_size, hidden_size, output_size=1, num_layers=2):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bias=False)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, _x):
            x, _ = self.lstm(_x)  # _x is input, size (seq_len, batch, input_size)
            s, b, h = x.shape  # x is output, size (seq_len, batch, hidden_size)
            x = x.view(s * b, h)
            x = self.fc(x)
            x = x.view(s, b, -1)  # 把形状改回来
            return x


    def predicate(model, test_x, test_y):
        test_x = test_x.type(torch.FloatTensor)
        test_y = test_y.type(torch.FloatTensor)
        model = model.eval()
        pre_test = model(test_x)
        pre_test = pre_test.view(-1).data.numpy()
        print('预测结果:{}\n'.format(pre_test))
        # print('实际结果:{}'.format(test_y))


    '''数据读取部分'''
    input = pd.read_excel("E:\SeaFile\Grade2_Tsinghua\华慧芯实习\Test_data_total\Total data.xlsx")
    input_array = input.to_numpy()
    input_array = np.delete(input_array, [0, 1, 2, 3], 1)

    output = pd.read_excel("E:\SeaFile\Grade2_Tsinghua\华慧芯实习\Test_data_total\label_new_total.xlsx")
    output_array = output.to_numpy()
    output_array = np.delete(output_array, [1, 2], 0)

    # 测试用例
    x_train, x_test, y_train, y_test = train_test_split(input_array.T, output_array.T, test_size=0.2, random_state=0)

    scaler = StandardScaler()  # 标准化转换
    scaler.fit(x_train)  # 训练标准化对象
    x_train = scaler.transform(x_train)  # 转换数据集

    scaler1 = StandardScaler()  # 标准化转换
    scaler1.fit(x_test)  # 训练标准化对象
    x_test = scaler1.transform(x_test)  # 转换数据集

    x_train = x_train.reshape(-1, 1, 3648)  # 3648是特征维度即光谱通道个数
    y_train = y_train.reshape(-1, 1, 1)

    x_test = x_test.reshape(-1, 1, 3648)  # 3648是特征维度
    y_test = y_test.reshape(-1, 1, 1)

    train_x = torch.from_numpy(x_train)
    train_y = torch.from_numpy(y_train)
    test_x = torch.from_numpy(x_test)
    test_y = torch.from_numpy(y_test)

    '''训练参数设置'''
    model = LSTM_Regression(3648, 8, output_size=1, num_layers=2)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    train_x = train_x.type(torch.FloatTensor)
    train_y = train_y.type(torch.FloatTensor)

    '''训练'''
    for i in range(1000):
        out = model(train_x)
        loss = loss_function(out, train_y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if (i + 1) % 100 == 0:
            print('Epoch: {}, Loss:{:.5f}'.format((i + 1), loss.item()))
            predicate(model, test_x, test_y)