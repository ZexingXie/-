#coding=utf-8
#Author:XingShell
from decision_stump import load_data
from decision_stump import decision_stump_multi_d,err_out_counter
import numpy as np
import time

min = 1e-10
class ModelStump:
    def __init__(self,determined_dimension,theta_result,s_result):
        self.determined_dimension,self.theta_result,self.s_result = determined_dimension,theta_result,s_result
    def predict(self,x):
        x = np.array(x)
        return (np.sign(x[:,self.determined_dimension]-self.theta_result+min))*self.s_result
    def accurate(self,x,y):
        e_out = np.where(self.predict(x) == y, 0, 1).sum() / np.size(y, 0)
        return e_out
    def errorLabels(self,x,y):
        idx = [i for i,x in enumerate(np.where(self.predict(x) == y, 0, 1)) if x==1]
        return idx

class AdaboostStumps:
    def __init__(self,a,basemodels):
        self.a,self.basemodels = a,basemodels

    def predict(self, x):
        result = 0
        for i,w in enumerate(self.a):
            result += w*self.basemodels[i].predict(x)
        return np.sign(result)


    def accurate(self, x, y):
        e_out = np.where(self.predict(x) == y, 0, 1).sum() / np.size(y, 0)
        return e_out

    def errorLabels(self, x, y):
        idx = [i for i, x in enumerate(np.where(self.predict(x) == y, 0, 1)) if x == 1]
        return idx


# def ModelError(x_datas,y_labels,ModelTree):
#     # 重新定制错误
#     errorIndex = []
#     for index,x in enumerate(x_datas):
#         ModelTree(x)

def AdaBoost(x_data, y_data,rounds=50):
    ms_list = []
    a = []
    N = len(y_data)
    W = [1/N]*N
    errorIndex = []
    determined_dimension, e_in_result, theta_result, s_result = decision_stump_multi_d(x_data, y_data,W)
    Model = (determined_dimension, theta_result, s_result)
    ms = ModelStump(*Model)
    errorIndex = ms.errorLabels(x_data, y_data)
    e_w = len(errorIndex)/len(y_data)
    for t in range(rounds):
        # 重新计算一遍错误向量
        errorIndex = ms.errorLabels(x_data,y_data)
        error_ratio = len(errorIndex)/len(y_data)
        if error_ratio == 0.5:
            return -1,None
        # print(e_w)
        # am = (1 / 2) * np.log((1 - error_ratio) / error_ratio)
        am = (1 / 2) * np.log((1 - e_w) / e_w)
        # if error_ratio<0.5:
        # print(errorIndex)
        a.append(am)
        ms_list.append(ms)
        am = np.abs(am)
        e_w = 0
        for i in range(len(y_data)):
            if i in errorIndex:
                W[i] *= np.exp(am)
                e_w += W[i]
            else:
                W[i] *= np.exp(-am)
        sum = np.sum(W)
        for index,w in enumerate(W):
            W[index] = w/sum
        # print(W)
        determined_dimension, e_in_result, theta_result, s_result = decision_stump_multi_d(x_data, y_data, W)
        Model = (determined_dimension, theta_result, s_result)
        ms = ModelStump(*Model)

    adaboost = AdaboostStumps(a,ms_list)
    e_in_result = adaboost.accurate(x_data,y_data)
    # print(a)
    print("E_IN:", e_in_result)

    return e_in_result,adaboost


if __name__ == '__main__':
    #开始时间
    start = time.time()
    x_train, y_train = load_data('data/train.txt')
    x_test, y_test = load_data('data/test.txt')
    _, Model = AdaBoost(x_train, y_train)
    # t = x_train,y_train
    # x_train, y_train = x_test,y_test
    # x_test, y_test = t
    yt = []
    ytest = []
    for r in range(1,1000):
        # Model = AdaBoost(x_train,y_train,r)
        y,Model = AdaBoost(x_train,y_train,r)

        if y==-1:
            break
        yt.append(y)
        ytest.append(Model.accurate(x_test,y_test))
    #
    import matplotlib.pyplot as plt  # 约定俗成的写法plt
    plt.plot(yt)
    plt.plot(ytest)
    plt.show()



    print("E_OUT:",Model.accurate(x_test,y_test))


