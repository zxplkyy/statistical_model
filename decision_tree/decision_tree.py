# -*- coding: utf-8 -*-
"""
   File Name：     decision_tree
   Description :
   Author :       zhengxinping
   date：          2019/3/7
"""
import cv2
import time
import logging
import numpy as np
import pandas as pd

total_class = 10

def log(func):
    def wrapper(*args,**kargs):
        start_time = time.time()
        logging.debug('start%s()'% func.__name__)
        ret = func(*args,**kargs)
        end_time = time.time()
        logging.debug("end %s(),cost %s seconds" %(func.__name__,end_time-start_time))
        return ret
    return wrapper

def binaryzation(img):
    #值转化为0或者1
    cv_img = img.astype(np.uint8)
    cv2.threshold(cv_img,50,1,cv2.cv.CV_THRESH_BINARY_INV,cv_img)
    return cv_img

@log
def binaryzation_features(trainset):
    features = []

    for img in trainset:
        img = np.reshape(img,(28,28))
        cv_img = img.astype(np.uint8)

        img_b = binaryzation(cv_img)
        features.append(img_b)
    features = np.array(features)
    features = np.reshape(features,(-1,784))

    return features

class Tree(object):
    """
    以特征作为根节点的树
    """
    def __init__(self,node_type,Class = None,feature = None):
        self.node_type = node_type
        self.dict ={}
        self.Class = Class
        self.feature = feature

    def add_tree(self,val,tree):
        self.dict[val] = tree

    def predict(self,features):
        """
        根据特征集，遍历这棵树
        :param features:
        :return:
        """
        if self.node_type == 'leaf':
            return self.Class

        tree = self.dict[features[self.feature]]
        return tree.predict(features)

def calc_ent(x):
    """
    calculate shanno ent of x x是一维，x[0]是样本数量
    :param x:
    :return:
    """
    x_val_list = set(x[i] for i in range(x.shape[0])) #统计类别总数
    ent = 0.0
    for x_val in x_val_list:
        p = float(x[x==x_val].shape[0])/x.shape[0]
        logp = np.log2(p)
        ent -= p*logp
    return ent

def calc_condition_ent(x,y):
    x_val_list = set(x[i] for i in range(x.shape[0]))
    ent = 0.0
    for x_value in x_val_list:

        sub_y = y[x==x_value]
        temp_ent = calc_ent(sub_y)
        ent += (float(sub_y.shape[0])/y.shape[0])*temp_ent
    return ent

def calc_ent_grap(x,y):
    """
    计算信息增益
    :param x:
    :param y:
    :return:
    """
    base_ent = calc_ent(y)
    condition_ent = calc_condition_ent(x,y)
    ent_grap = base_ent - condition_ent
    return  ent_grap



def recurse_train(train_set,train_label,features,epsilon):
    """

    :param train_set:
    :param train_label: 标签集
    :param features: 特征集
    :param epsilon: 阈值
    :return:
    """
    global total_class
    LEAF = 'leaf'
    INTERNAL = "internal"

    #1. 若所有实例都属于同一类，则T为单节点树
    label_set = set(train_set) #因为set中的元素是不重复的，当长度为1时，意味着所有元素的标签都一样
    if len(label_set)==1:
        return Tree(LEAF,Class=label_set.pop())

    #2. 若特征集为空,T为单节点树，选择实例数最大的类作为该节点的类标记

    (max_class,max_len) = max([(i,len(filter(lambda x:x==i,train_label)))for i in range(total_class)],key = lambda x:x[1])

    if len(features)==0:
        return Tree(LEAF,Class=max_class)
    # 3. 计算每一种特征对信息的增益，选出影响力最大的特征
    max_feature = 0
    max_gda = 0
    D = train_label
    HD = calc_ent(D)
    for feature in features:
        A = np.array(train_set[:,feature].flat) #提取出某一条特征
        gda = HD - calc_condition_ent(A,D)
        if gda > max_gda:
            max_gda,max_feature = gda,feature
    #4. 若最大的信息增益仍比阈值小，那么就选择实例数最大的类作为该节点的类标记
    if max_gda < epsilon:
        return Tree(LEAF,Class=max_class)
    #5. 构建非空子集
    sub_features = filter(lambda x: x!= max_feature,features) #获取子特征
    tree = Tree(INTERNAL,feature = max_feature)

    feature_col = np.array(train_set[:,max_feature].flat)
    feature_val_list = set([feature_col[i] for i in range(feature_col.shape[0])]) #取出某一列特征的所有取值
    for feature_value in feature_val_list:
        # 根据该特征的不同取值，划分出子集
        index = []
        for i in range(len(train_label)):
            if train_set[i][max_feature] == feature_value: #若某一实例存在该特征，那么就添加该实例
                index.append(i)
        sub_train_set = train_set[index]
        sub_train_label = train_label[index]

        sub_tree = recurse_train(sub_train_set,sub_train_label,sub_features,epsilon)
        tree.add_tree(feature_value,sub_tree)
    return tree

@log
def train(train_set,train_label,features,epsilon):
    return recurse_train(train_set,train_label,features,epsilon)
@log
def predict(test_set,tree):
    result = []
    for features in test_set:
        tmp_predict = tree.predict(features)
        result.append(tmp_predict)
    return np.array(result)

