import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
def dataload(spilt_size):
    '''数据读取部分'''
    input = pd.read_excel("E:\SeaFile\Grade2_Tsinghua\华慧芯实习\Test_data_total\Total data.xlsx")
    input_array = input.to_numpy()
    input_array = np.delete(input_array, [0, 1, 2, 3], 1)

    output = pd.read_excel("E:\SeaFile\Grade2_Tsinghua\华慧芯实习\Test_data_total\label_new_total.xlsx")
    output_array = output.to_numpy()
    output_array = np.delete(output_array, [1, 2], 0)

    # 测试用例
    x_train, x_test, y_train, y_test = train_test_split(input_array.T, output_array.T, test_size=spilt_size, random_state=0)

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

    train_x = torch.from_numpy(x_train)  # 转换为Tensor
    train_y = torch.from_numpy(y_train)
    test_x = torch.from_numpy(x_test)
    test_y = torch.from_numpy(y_test)
    return train_x,train_y,test_x,test_y