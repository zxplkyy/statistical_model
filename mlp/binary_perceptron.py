# -*- coding: utf-8 -*-
"""
   File Name：     binary_perceptron
   Description :  sigmoid 函数，二分法
   Author :       zhengxinping
   date：          2019/3/7
"""
import random
import pandas as pd
import numpy as np
import cv2
import time
from sklearn.metrics import accuracy_score
def get_hog_features(trainset):
    features =[]
    hog = cv2.HOGDescriptor('../hog.xml')
    for img in trainset:
        img = np.reshape(img,(28,28))
        cv_img = img.astype(np.uint8)
        hog_feature = hog.compute(cv_img)
        features.append(hog_feature)
    features = np.array(features)
    features = np.reshape(features,(-1,324))
    return features


class Perceptron(object):

    def __init__(self):
        self.learning_rate = 0.0001
        self.max_iteration = 5000

    def predict_(self,x):
        wx = sum([self.W[j]*x[j] for j in range(len(self.W))])
        return int(wx > 0)

    def train(self,features,labels):
        self.w = [0.0] * (len(features[0])+1) #多一列的原因是因为有b
        correct_count = 0
        time = 0
        while time < self.max_iteration:
            index = random.randint(0,len(labels)-1) #任意取一个样本
            x = list(features[index])
            x.append(1.0) #添加偏置b
            y = 2 * labels[index]-1 #y是0时，转换为-1
            wx = sum([self.w[j]* x[j] for j in range(len(self.w))])
            if wx*y > 0:
                correct_count += 1
                if correct_count > self.max_iteration:
                    break
                continue
            for i in range(len(self.w)):
                self.w[i] += self.learning_rate *(y*x[i]) #参数发生变化

    def predict(self,features):
        labels = []
        for feature in features:
            x = list(feature)
            x.append(1)
            labels.append(self.predict_(x))
        return labels

if __name__ == '__main__':
    print("Start read data")
    time1 = time.time()
    raw_data1 = pd.read_csv("../dataset/train.csv",header = 0)
    raw_data2 = pd.read_csv("../dataset/test.csv",header = 0)
    train_data = raw_data1.values
    test_data = raw_data2.values

    train_features = get_hog_features(train_data[0::,1::]) #获取所有行，第一列到最后一列
    train_labels = train_data[::,0]  #获取第一列

    test_features = get_hog_features(test_data[0::,1::])
    test_labels = test_data[::,0]

    time2 = time.time()
    print("read data cost '",time2-time1,'second','\n')
    print("start training")
    p = Perceptron()
    p.train(train_features,train_labels)

    time3 = time.time()
    print("training cost",time3 - time2,'second','\n')

    print("start predicting")
    test_predict = p.predict(test_features)
    time4 = time.time()
    print("predict cost", time4 - time3,"second","\n")

    score = accuracy_score(test_labels,test_predict)
    print("The accuracy is",score)






