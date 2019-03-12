# -*- coding: utf-8 -*-
"""
   File Name：     naive_bayes
   Description :
   Author :       zhengxinping
   date：          2019/3/7
"""
import numpy as np
import cv2

class_num = 10
feature_len = 784 # 28*28 每个图片的像素作为一个特征
def binaryzation(img):
    cv_img = img.astype(np.uint8)
    cv2.threshold(cv_img,50,1,cv2.cv.CV_THRESH_BINARY_INV,cv_img)
    return cv_img

def train(trainset,train_labels):
    prior_probability = np.zeros(class_num) #某一类出现的概率
    conditional_probability = np.zeros((class_num,feature_len,2)) #表示在某一类别的前提下，某个特征在某个取值下的概率

    #计算先验概率及条件概率
    for i in range(len(train_labels)):
        img = binaryzation(trainset[i])
        label = train_labels[i]

        prior_probability[label] += 1
        for j in range(feature_len):
            conditional_probability[label][j][img[j]] += 1 #第label个类下，第j个特征的取值数量加1

    for i in range(class_num):
        for j in range(feature_len):
            pix_0 = conditional_probability[i][j][0]
            pix_1 = conditional_probability[i][j][1]

            #计算0，1 像素点对应的条件概率
            probability_0 = (float(pix_0)/float(pix_0+pix_1))*1000000 + 1
            probability_1 = (float(pix_1)/float(pix_0+pix_1))*1000000 + 1

            conditional_probability[i][j][0] = probability_0
            conditional_probability[i][j][1] = probability_1

    return prior_probability,conditional_probability

def calculate_probability(img,label,prior_probability,conditional_probability):
    """
    计算某张图片属于某个标签的概率
    :param img:
    :param label:
    :return:
    """
    probability = int(prior_probability[label])
    for i in range(len(img)):
        probability *= int(conditional_probability[label][i][img[i]])
    return probability



def predict(testset,prior_probability,conditional_probability):
    predict = []
    for img in testset:
        img = binaryzation(img)
        max_label = 0
        max_probability = calculate_probability(img,0,prior_probability,conditional_probability)
        for j in range(1,10):
            probability = calculate_probability(img,j,prior_probability,conditional_probability)
            if max_probability < probability:
                max_label = j
                max_probability = probability
        predict.append(max_label)
    return np.array(predict)