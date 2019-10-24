# -*- coding:utf-8 -*-
# Author: Xing Shell
import numpy as np

def load_data(file_name):
    x = []
    y = []
    with open(file_name, 'r+') as f:
        for line in f:
            line = line.rstrip("\n")
            temp = line.split(" ")
            temp.insert(0, '1')
            x_temp = [float(val) for val in temp[:-1]]
            y_tem = [int(val) for val in temp[-1:]][0]
            x.append(x_temp)
            y.append(y_tem)

    nx = np.array(x)
    ny = np.array(y)
    return nx, ny

def sign_zero_as_neg(x):
    """
    这里修改了np自带的sign函数，当传入的值为0的时候，不再返回0，而是-1；
    也就是说在边界上的点按反例处理
    :param x:
    :return:
    """
    result = np.sign(x)
    result[result == 0] = -1
    return result


def err_in_counter(x_arr, y_arr, s, theta=None,errorWeightofData=None):
    # 原作者使用theta显得非常多余
    theta = np.tile(x_arr,(x_arr.shape[0],1)).T
    x_arr = theta.T
    y_arr = np.tile(y_arr,(y_arr.shape[0],1))
    result = s * sign_zero_as_neg(x_arr - theta)
    if errorWeightofData is None:
        err_tile = np.where(result == y_arr, 0, 1).sum(1)
    else:
        errorWeightofData = np.array(errorWeightofData)

        err_tile = np.dot(np.where(result == y_arr, 0, 1),errorWeightofData)
    return err_tile.min(), err_tile.argmin()


def err_out_counter(x_arr, y_arr, s, theta, dimension):
    temp = s * sign_zero_as_neg(x_arr.T[dimension] - theta)
    e_out = np.where(temp == y_arr, 0, 1).sum() / np.size(x_arr, 0)
    return e_out


def decision_stump_1d(x_arr, y_arr,errorWeightofData=None):
    theta = x_arr
    # 原作者的屠龙技
    # theta_tile = np.tile(theta, (len(x_arr), 1)).T
    # x_tile = np.tile(x_arr, (len(theta), 1))
    # y_tile = np.tile(y_arr, (len(theta), 1))
    # err_pos, index_pos = err_in_counter(x_tile, y_tile, 1, theta_tile)
    # err_neg, index_neg = err_in_counter(x_tile, y_tile, -1, theta_tile)
    err_pos, index_pos = err_in_counter(x_arr,y_arr,1,errorWeightofData=errorWeightofData)
    err_neg, index_neg = err_in_counter(x_arr,y_arr,-1,errorWeightofData=errorWeightofData)
    if err_pos < err_neg:
        return err_pos / len(y_arr), index_pos, 1
    else:
        return err_neg / len(y_arr), index_neg, -1


def decision_stump_multi_d(x, y,errorWeightofData=None):
    x = x.T
    dimension, e_in, theta, s = 0, float('inf'), 0, 0
    for i in range(np.size(x, 0)):
        e_in_temp, index, s_temp = decision_stump_1d(x[i], y,errorWeightofData=errorWeightofData)
        if e_in_temp < e_in:
            dimension, e_in, theta, s = i, e_in_temp, x[i][index], s_temp
        # 错误率相等的时候随机选择
        if e_in_temp == e_in:
            pick_rate = np.random.uniform(0, 1)
            if pick_rate > 0.5:
                dimension, e_in, theta, s = i, e_in_temp, x[i][index], s_temp
    return dimension, e_in, theta, s


if __name__ == '__main__':
    x_train, y_train = load_data('data/train.txt')
    x_test, y_test = load_data('data/test.txt')

    determined_dimension, e_in_result, theta_result, s_result = decision_stump_multi_d(x_train, y_train)
    print("E_IN:", e_in_result)
    print("E_OUT:", err_out_counter(x_test, y_test, s_result, theta_result, determined_dimension))