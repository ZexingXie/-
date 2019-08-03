"""
实现代码主要参考了https: // github.com / Dod - o / Statistical - Learning - Method_Code / blob / master / SVM / SVM.py
感谢前人的工作
"""
import numpy as np
import math
class SVM:
    def __init__(self, x, y, C = 1, toler = 0):
        self.trainData_x = np.array(x)  # 训练数据集
        self.trainLabel = np.array(y)  # 训练标签集
        self.m, self.n = np.shape(self.trainData_x)  # m：训练集数量    n：样本特征数目
        self.C = C  # 惩罚参数
        self.toler = toler  # 松弛变量
        self.b = 0  # SVM中的偏置b
        self.alpha = [1] * self.m   # α 非supportVec 为0
        self.supportVecIndex = []
        self.E = [0 * self.trainLabel[i] for i in range(self.trainLabel.shape[0])]  # SMO运算过程中的Ei

    def train(self, iter=100):
        # iterStep：迭代次数，超过设置次数还未收敛则强制停止
        # parameterChanged：单次迭代中有参数改变则增加1
        iterStep = 0;
        parameterChanged = 1

        # 如果没有达到限制的迭代次数以及上次迭代中有参数改变则继续迭代
        # parameterChanged==0时表示上次迭代没有参数改变，如果遍历了一遍都没有参数改变，说明
        # 达到了收敛状态，可以停止了
        while (iterStep < iter) and (parameterChanged > 0):
            # 打印当前迭代轮数
            print('iter:%d:%d' % (iterStep, iter))
            # 迭代步数加1
            iterStep += 1
            # 新的一轮将参数改变标志位重新置0
            parameterChanged = 0
            # 大循环遍历所有样本，用于找SMO中第一个变量
            for i in range(self.m):
                if self.isSatisfyKKT(i) == False:
                    # 第一个变量α的下标i已经确定，接下来按照“7.4.2 变量的选择方法”第二步
                    # 选择变量2。由于变量2的选择中涉及到|E1 - E2|，因此先计算E1
                    E1 = self.calcEi(i)

                    # 选择第2个变量
                    E2, j = self.getAlphaJ(E1, i)

                    # 参考“7.4.1两个变量二次规划的求解方法” P126 下半部分
                    # 获得两个变量的标签
                    y1 = self.trainLabel[i]
                    y2 = self.trainLabel[j]
                    # 复制α值作为old值
                    alphaOld_1 = self.alpha[i]
                    alphaOld_2 = self.alpha[j]
                    # 依据标签是否一致来生成不同的L和H
                    if y1 != y2:
                        L = max(0, alphaOld_2 - alphaOld_1)
                        H = min(self.C, self.C + alphaOld_2 - alphaOld_1)
                    else:
                        L = max(0, alphaOld_2 + alphaOld_1 - self.C)
                        H = min(self.C, alphaOld_2 + alphaOld_1)
                    # 如果两者相等，说明该变量无法再优化，直接跳到下一次循环
                    if L == H:   continue

                    # 计算α的新值
                    # 依据“7.4.1两个变量二次规划的求解方法”式7.106更新α2值
                    # 先获得几个k值，用来计算事7.106中的分母η
                    k11 = self.Kernel(i,j)
                    k22 = self.Kernel(j,j)
                    k21 = self.Kernel(j,i)
                    k12 = self.Kernel(i,j)
                    # 依据式7.106更新α2，该α2还未经剪切
                    alphaNew_2 = alphaOld_2 + y2 * (E1 - E2) / (k11 + k22 - 2 * k12)
                    # 剪切α2
                    if alphaNew_2 < L:
                        alphaNew_2 = L
                    elif alphaNew_2 > H:
                        alphaNew_2 = H
                    # 更新α1，依据式7.109
                    alphaNew_1 = alphaOld_1 + y1 * y2 * (alphaOld_2 - alphaNew_2)

                    # 依据“7.4.2 变量的选择方法”第三步式7.115和7.116计算b1和b2
                    b1New = -1 * E1 - y1 * k11 * (alphaNew_1 - alphaOld_1) \
                            - y2 * k21 * (alphaNew_2 - alphaOld_2) + self.b
                    b2New = -1 * E2 - y1 * k12 * (alphaNew_1 - alphaOld_1) \
                            - y2 * k22 * (alphaNew_2 - alphaOld_2) + self.b

                    # 依据α1和α2的值范围确定新b
                    if (alphaNew_1 > 0) and (alphaNew_1 < self.C):
                        bNew = b1New
                    elif (alphaNew_2 > 0) and (alphaNew_2 < self.C):
                        bNew = b2New
                    else:
                        bNew = (b1New + b2New) / 2

                    # 将更新后的各类值写入，进行更新
                    self.alpha[i] = alphaNew_1
                    self.alpha[j] = alphaNew_2
                    self.b = bNew

                    self.E[i] = self.calcEi(i)
                    self.E[j] = self.calcEi(j)

                    # 如果α2的改变量过于小，就认为该参数未改变，不增加parameterChanged值
                    # 反之则自增1
                    if math.fabs(alphaNew_2 - alphaOld_2) >= 0.00001:
                        parameterChanged += 1

                # 打印迭代轮数，i值，该迭代轮数修改α数目
                print("iter: %d i:%d, pairs changed %d" % (iterStep, i, parameterChanged))

                # 全部计算结束后，重新遍历一遍α，查找里面的支持向量
            for i in range(self.m):
                # 如果α>0，说明是支持向量
                if self.alpha[i] > 0:
                    # 将支持向量的索引保存起来
                    self.supportVecIndex.append(i)

    def isSatisfyKKT(self, i):
        '''
        查看第i个α是否满足KKT条件
        :param i:α的下标
        '''
        gxi =self.calc_gxi(i)
        yi = self.trainLabel[i]


        if (math.fabs(self.alpha[i]) < self.toler) and (yi * gxi >= 1):
            return True

        elif (math.fabs(self.alpha[i] - self.C) < self.toler) and (yi * gxi <= 1):
            return True

        elif (self.alpha[i] > -self.toler) and (self.alpha[i] < (self.C + self.toler)) \
                and (math.fabs(yi * gxi - 1) < self.toler):
            return True

        return False

    def calc_gxi(self, i):
        gxi = 0
        index = [i for i, alpha in enumerate(self.alpha) if alpha != 0]
        # 遍历每一个非零α，i为非零α的下标
        for j in index:
            # 计算g(xi)
            gxi += self.alpha[j] * self.trainLabel[j] * self.Kernel(i,j)
        # 求和结束后再单独加上偏置b
        gxi += self.b
        # 返回
        return gxi

    def Kernel(self,x1,x2):
        sum = 0
        for x_1 in range(self.n):
            sum += self.trainData_x[x1][x_1]*self.trainData_x[x2][x_1]
        return sum
    def calcEi(self, i):
        gxi = self.calc_gxi(i)
        return gxi - self.trainLabel[i]
    def getAlphaJ(self, E1, i):
        '''
        SMO中选择第二个变量
        :param E1: 第一个变量的E1
        :param i: 第一个变量α的下标
        :return: E2，α2的下标
        '''
        #初始化E2
        E2 = 0
        #初始化|E1-E2|为-1
        maxE1_E2 = -1
        #初始化第二个变量的下标
        maxIndex = -1

        #这一步是一个优化性的算法
        #实际上书上算法中初始时每一个Ei应当都为-yi（因为g(xi)由于初始α为0，必然为0）
        #然后每次按照书中第二步去计算不同的E2来使得|E1-E2|最大，但是时间耗费太长了
        #作者最初是全部按照书中缩写，但是本函数在需要3秒左右，所以进行了一些优化措施
        #--------------------------------------------------
        #在Ei的初始化中，由于所有α为0，所以一开始是设置Ei初始值为-yi。这里修改为与α
        #一致，初始状态所有Ei为0，在运行过程中再逐步更新
        #因此在挑选第二个变量时，只考虑更新过Ei的变量，但是存在问题
        #1.当程序刚开始运行时，所有Ei都是0，那挑谁呢？
        #   当程序检测到并没有Ei为非0时，将会使用随机函数随机挑选一个
        #2.怎么保证能和书中的方法保持一样的有效性呢？
        #   在挑选第一个变量时是有一个大循环的，它能保证遍历到每一个xi，并更新xi的值，
        #在程序运行后期后其实绝大部分Ei都已经更新完毕了。下方优化算法只不过是在程序运行
        #的前半程进行了时间的加速，在程序后期其实与未优化的情况无异
        #------------------------------------------------------

        #获得Ei非0的对应索引组成的列表，列表内容为非0Ei的下标i
        nozeroE = [i for i, Ei in enumerate(self.E) if Ei != 0]
        #对每个非零Ei的下标i进行遍历
        for j in nozeroE:
            #计算E2
            E2_tmp = self.calcEi(j)
            #如果|E1-E2|大于目前最大值
            if math.fabs(E1 - E2_tmp) > maxE1_E2:
                #更新最大值
                maxE1_E2 = math.fabs(E1 - E2_tmp)
                #更新最大值E2
                E2 = E2_tmp
                #更新最大值E2的索引j
                maxIndex = j
        #如果列表中没有非0元素了（对应程序最开始运行时的情况）
        if maxIndex == -1:
            maxIndex = i
            while maxIndex == i:
                #获得随机数，如果随机数与第一个变量的下标i一致则重新随机
                maxIndex = int(np.random.uniform(0, self.m))
            #获得E2
            E2 = self.calcEi(maxIndex)

        #返回第二个变量的E2值以及其索引
        return E2, maxIndex
    def w(self):
        w = np.zeros(self.m)
        for i in self.supportVecIndex:
            # 遍历所有支持向量，计算求和式
            # 对每一项子式进行求和，最终计算得到求和项的值
            w += self.alpha[i] * self.trainLabel[i] * self.trainData_x[i]
        return w
    def print(self):
        w = self.w()
        mm = 0
        for frac in w:
            mm += frac**2
        w /= (mm**(0.5))
        print(w)
        print('b',self.b)


if __name__ == '__main__':
    x = [[-1,-1],[1,1]]
    y = [-1,1]
    a = SVM(x,y)
    a.train()
    a.print()

