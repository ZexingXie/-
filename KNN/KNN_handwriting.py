from utils import knn,errorTest as e
import imread_vec
import numpy as np
from utils import Norm as n


if __name__ == '__main__':
    dir_memory = './trainingDigits/'
    M,L = imread_vec.blockMemory(dir_memory)
    M,a,b = n.autoNorm(M)
    model = knn.knn(M,L,k=3)
    dir_test = './testDigits/'
    testM,testL = imread_vec.blockMemory(dir_test)
    testM = n.Norm(testM,a,b)
    y_predict = []
    for i,test in enumerate(testM):
        y = model(test)
        y_predict.append(y)
    accurate_ = e.accurate(y_true=testL, y_pred=y_predict)
    print(accurate_)

    # k =3
    # 0.9873
    # 0.985  -- 归一
    # k = 5
    # 0.9344
    # 0.978  -- 归一
