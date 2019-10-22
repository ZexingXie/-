from math import log
from collections import Counter
import operator
def createDataSet1():    # 创造示例数据
    dataSet = [['长', '粗', '男'],
               ['短', '粗', '男'],
               ['短', '粗', '男'],
               ['长', '细', '女'],
               ['短', '细', '女'],
               ['短', '粗', '女'],
               ['长', '粗', '女'],
               ['长', '粗', '女']]
    labels = ['头发','声音']  #两个特征
    return dataSet,labels

def SameAs(labels,dataSet):
    for index,label in enumerate(labels):
        attrs = [l[index] for l in dataSet]
        if len(attrs) != attrs.count(attrs[0]):
            # 只要有一个划分能划分，不代表是否有意义
            return False
    return True


# def choose(dataSet,labels):
#     import numpy as np
#     return np.random.random_integers(0,len(labels)-1)
def calculate(dataSet):
    labels = []
    for y in dataSet:
        labels.append(y[-1])
    Len = len(labels)
    Counter_labels = Counter(labels)
    baseEntropy = 0
    for y, n in Counter_labels.items():
        prob = n / Len
        baseEntropy += -prob * log(prob, 2)
    return baseEntropy
def choose_gain(dataSet,labels):
    # 当前信息熵
    bestInfoGain = 0
    bestAttrIndex = 0
    baseEntropy = calculate(dataSet)
    newEntropy = 0
    for index in range(len(labels)):
        featList = [example[index] for example in dataSet]
        uniqueVals = set(featList)
        for value in uniqueVals:
            dataSub = splitDataSet(dataSet,value,index)
            prob = len(dataSub) / float(len(dataSet))
            newEntropy += prob*calculate(dataSub)
        if newEntropy > bestInfoGain:
            bestInfoGain = newEntropy
            bestAttrIndex = index
    return bestAttrIndex


    for attr in labels:
        # 分别计算每个属性的gain指数
        pass


def splitDataSet(dataSet,value,attr_index):
    # 返回 attr_index==value的dataSet，并且已经去掉该attr



def splitDataSet(dataSet,value,attr_index):
    retDataSet = []
    for featVec in dataSet:
        if featVec[attr_index] == value:
            reducedFeatVec = featVec[:attr_index]
            reducedFeatVec.extend(featVec[attr_index + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet



def createTree(dataSet,labels):
    # 如果样本已经是同一个类别无需划分
    yList = [y[-1] for y in dataSet]
    if yList.count(yList[0]) == len(yList):
        return yList[0]
    # 如果无法划分，即labels为空或每个样本在属性上性质相同
    elif len(labels)==0 or SameAs(labels,dataSet):
        return Counter(yList).most_common(1)[0][0]
    else:
        attr_index = choose(dataSet,labels)
        myTree = {labels[attr_index]: {}}  # 分类结果以字典形式保存
        featValues=[example[attr_index] for example in dataSet]
        uniqueVals=set(featValues)

        for value in uniqueVals:
            # 根据属性划分
            subDataSet = splitDataSet(dataSet, value, attr_index) # 选出该属性划分数据
            if len(subDataSet)==0:
                # 因为按照大的数据集合，不能保证含有该属性值是否已经被划分包含
                # 使用父节点的分布作为先验分布
                return Counter([y[-1] for y in dataSet]).most_common(1)[0][0]
            else:
                temsave = labels.pop(attr_index)  # 该属性被用掉
                node = createTree(subDataSet,labels)
                labels.insert(attr_index,temsave)
                myTree[temsave][value] = node
        return myTree

if __name__=='__main__':
    dataSet, labels=createDataSet1()  # 创造示列数据
    print(createTree(dataSet, labels))  # 输出决策树模型结果


