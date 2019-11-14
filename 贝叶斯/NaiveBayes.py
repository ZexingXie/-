import numpy as np
import math
def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)
    pAbusive = np.sum(trainCategory)/len(trainCategory)
    # p0Denom = 0.0; p1Denom = 0.0
    # p0Num = np.zeros(len(trainMatrix[0]))
    # p1Num = np.zeros(len(trainMatrix[0]))
    p0Num = np.ones(len(trainMatrix[0]))
    p1Num = np.ones(len(trainMatrix[0]))
    for i in range(numTrainDocs):
        if trainCategory[i] == 0:
            p0Num += trainMatrix[i]
        else:
            p1Num += trainMatrix[i]
    p0Num /= 2+np.sum(p0Num)
    p1Num /= 2+np.sum(p1Num)
    p0Num = np.log(p0Num)
    p1Num = np.log(p1Num)
    return pAbusive,p0Num,p1Num

def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    p1 = np.sum(vec2Classify*p1Vec) + np.log(pClass1)
    p0 = np.sum(vec2Classify * p0Vec) + np.log(1.0-pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


if __name__ == '__main__':
    from create_data import *
    data, Label = loadDataSet()
    vocabList = creaseVocabList(data)
    trainMat = []
    for postinDoc in data:
        trainMat.append(setOfWords2Vec(vocabList,postinDoc))
    pAb,p0v,p1v = trainNB0(trainMat,Label)
    print(p1v)
    # arg = np.argmax(p1v)
    import operator
    dict = {}
    for key,value in zip(vocabList,p1v):
        dict[key] = value
    ans = sorted(dict.items(),key=operator.itemgetter(1),reverse=True)
    print(ans)