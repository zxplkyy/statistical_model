# -*- coding: utf-8 -*-
"""
   File Name：     max_entropy
   Description :
   Author :       zhengxinping
   date：          2019/3/8
"""

from collections import defaultdict
import math
class MaxEnt(object):
    def init_params(self,X,Y):
        self.X_ = X
        self.Y_ = set()

        self.cal_Pxy_Px(X,Y)

        self.N = len(X) #number of training set
        self.n = len(self.Pxy) #标签特征对的数量
        self.M = 10000 #认为是学习速率

        self.build_dict()
        self.cal_EPxy() #计算(x,y)的期望

    def build_dict(self):
        """
        将特征标签对加上其对应的索引
        :return:
        """
        self.id2xy = {}
        self.xy2id = {}

        for i,(x,y) in enumerate(self.Pxy):
            self.id2xy[i] = (x,y)
            self.xy2id[(x,y)] = i

    def cal_Pxy_Px(self,X,Y):
        """
        统计(x,y)的出现的次数，以及x出现的次数
        :param X:
        :param Y:
        :return:
        """
        self.Pxy = defaultdict(int) #存放特征标签对对应的数量
        self.Px = defaultdict(int) #存放特征支持的样本数

        for i in range(len(X)):
            x_,y = X[i],Y[i] #获取每一个样本
            self.Y_.add(y)

            for x in x_:
                self.Pxy[(x,y)] += 1 #统计特征标签对的数量
                self.Px[x] += 1 #统计某一特征的支持的样本数


    def fxy(self,x,y):
        # 特征函数，若(x,y)存在，则返回1，否则返回0

        return (x,y) in self.xy2id

    def cal_EPxy(self):
        # 计算特征函数f(x,y)关于经验分布P(X,Y)的期望,因为f(x,y)的取值不是0 就是1，所以当f(x,y)存在时，期望就为(x,y)出现的概率
        self.EPxy = defaultdict(float)
        for id in range(self.n):
            (x,y) = self.id2xy[id]
            self.EPxy[id] = float(self.Pxy[(x,y)])/float(self.N)



    def cal_pyx(self,X,y):
        #计算某个样本的所有特征在某一标签下的概率之和，即在某一标签确定的情况下，样本出现这些特征的概率总和
        result = 0.0
        for x in X:
            if self.fxy(x,y):
                id = self.xy2id[(x,y)]
                result += self.w[id]
        return (math.exp(result),y)


    def cal_probality(self,X):
        #理解为某一样本属于某一标签的概率
        Pyxs = [(self.cal_pyx(X,y))for y in self.Y_] #个数为标签的种类数
        Z = sum([prob for prob,y in Pyxs])
        return [(prob/Z,y) for prob, y in Pyxs]

    def cal_EPx(self):
        """
        :return:
        """
        self.EPx = [0.0 for i in range(self.n)]
        for i,X in enumerate(self.X_): #计算某一个样本
            Pyxs = self.cal_probality(X) #计算X属于各个标签的概率
            for x in X:
                for Pyx,y in Pyxs:
                    if self.fxy(x,y):
                        id = self.xy2id[(x,y)]
                        self.EPx[id] += Pyx*(1.0/self.N)



    def train(self,X,Y):
        self.init_params(X,Y)
        self.w = [0.0 for i in range(self.n)] #有多少个特征函数，就有多少个约束条件
        max_iterater = 1000
        for times in range(max_iterater):
            sigmas = []
            self.cal_EPx() #计算各个特征x的先验概率
            for i in range(self.n):
                sigma = 1/self.M * math.log(self.EPxy[i]/self.EPx[i])
                sigmas.append(sigma)
            self.w = [self.w[i] + sigmas[i] for i in range(self.n)] #对w进行更新

    def predict(self,testset):
        results = []
        for test in testset:
            result = self.cal_probality(test)
            results.append(max(result,key=lambda x:x[0])[1])
        return results




def rebuild_features(features):
    """
    将之前feature的(a0,a1,a2,a3,...)变成
    （0_a0,1_a1,2_a2,3_a3,...）
    :param features:
    :return:
    """
    new_features = []
    for feature in features:
        new_feature = []
        for i,f in enumerate(feature):
            new_feature.append(str(i)+"_"+str(f))
        new_features.append(new_feature)
    return new_features
