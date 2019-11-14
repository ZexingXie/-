
import numpy as np
import random
from create_data import *
from NaiveBayes import *

def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W*',bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2 ]

def spamTest():
    docList = [];classList = [];
    for i in range(1,26):
        wordList = textParse(open('email/ham/%d.txt'%i,encoding='ISO-8859-1').read())
        docList.append(wordList)
        classList.append(0)
        wordList = textParse(open('email/spam/%d.txt'%i,encoding='ISO-8859-1').read())
        docList.append(wordList)
        classList.append(1)

    vocabList = creaseVocabList(docList)
    trainingSet = list(range(0,50));testSet = []

    for i in range(10):
        randIndex = int(random.randint(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        trainingSet.pop(randIndex)
    trainMat = [];trainClasses=[]

    for docIndex in trainingSet:
        trainMat.append(bagOfWords2Vec(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    pSpam, p0V, p1V = trainNB0(array(trainMat), np.array(trainClasses))
    import operator
    dict = {}
    for key, value in zip(vocabList, p1V):
        dict[key] = value
    ans = sorted(dict.items(), key=operator.itemgetter(1), reverse=True)
    print(ans)
    errorCount = 0
    for docIndex in testSet:  # classify the remaining items
        wordVector = bagOfWords2Vec(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print(errorCount/len(testSet))

if __name__ == '__main__':

    spamTest()

# f = open('email/ham/6.txt',encoding='ISO-8859-1').read()
# print(f)