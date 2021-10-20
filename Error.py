import pandas as pd
from math import sqrt

class pre_mis():
    '''该类实现计算误差函数'''

    def __init__(self, pre_data, true_data):
        self.pre_data = pre_data
        self.true_data = true_data

    def pre_RMSE(self):
        dis = self.pre_data - self.true_data
        RMSEP = sqrt(sum(sum(dis * dis))) / len(self.pre_data)
        return RMSEP

    def pre_Rp(self):
        A = pd.Series(self.pre_data)
        B = pd.Series(self.true_data)
        Rp = B.corr(A, method='pearson')
        return Rp