# 实现滑动窗口 GPR
import numpy as np
from models.MyGP import *

class WindowGPR:
    """
    实现滑动窗口机制的GPR
    GPR 模型本身可以是库函数中的，也可以是自己实现的（理论上也可以是任意的实现了 __call__ 的对象）
    Parameters:
        model: GPR 模型
        window_size: 滑动窗口大小, 即 每一步向前看多少步
        step_size: 滑动步长    
    """
    def __init__(self, model, window_size, step_size):  
        self.window_size = window_size  
        self.step_size = step_size  
        self.gpr = model  

    def fit(self, X, y): 
        # fit 时，只需要保存数据，不需要训练  
        self.X = X  
        self.y = y  
        # self.gpr.fit(X[:self.window_size], y[:self.window_size])   

    def predict(self, X):
        y_pred = []
        # 每一步预测，使用与它最近的 window_size 个数据进行训练
        current_x = self.X[-self.window_size:]
        current_y = self.y[-self.window_size:]

        for i in range(0, len(X), self.step_size):

            offset = 0
            if i + self.window_size > len(X):
                offset = i + self.window_size - len(X)

            x = X[i:i+self.window_size - offset]   
            y = self.gpr.predict(current_x, current_y, x)
            y_pred.append(y)
            # 更新数据
            current_x = np.concatenate([current_x, x])
            current_y = np.concatenate([current_y, y])
            current_x = current_x[self.step_size:]
            current_y = current_y[self.step_size:]

        return np.concatenate(y_pred)

    def score(self, X, y):
        return self.gpr.score(X, y)


