# -*- coding: utf-8 -*-
"""
   File Name：     knn
   Description :
   Author :       zhengxinping
   date：          2019/3/7
"""
import cv2
import numpy as np

k = 10
def get_hog_features(trainset):
    """
    提取hog特征
    :param trainset:
    :return:
    """
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

def predict(testset,trainset,train_labels):
    predict = []
    count = 0
    for test_vec in testset:
        print(count)
        count+=1

        knn_list = [] #当前k个最近的邻居
        max_index = -1 #当前k个最近邻居中最远点的坐标
        max_dist = 0 # 当前k个最近邻居中距离最远点的距离

        #前k个点
        for i in range(k):
            label = train_labels[i]
            train_vec = trainset[i]
            dist = np.linalg.norm(train_vec-test_vec) #计算两个点的欧式距离
            knn_list.append((dist,label))
        # 剩下的点
        for i in range(k,len(train_labels)):
            label = train_labels[i]
            train_vec = trainset[i]
            dist = np.linalg.norm(train_vec-test_vec)

            if max_index < 0:
                for j in range(k):
                    if max_dist < knn_list[j][0]:
                        max_index = j
                        max_dist = knn_list[j][0]

            # 若当前点距离比最远点还小，则替换
            if dist < max_dist:
                knn_list[max_index] = (dist,label)
                max_index = -1
                max_dist = 0
        # 统计选票
        class_total = 10
        class_count = [0 for i in range(class_total)]
        for dist,label in knn_list:
            class_count[label] += 1

        mmax = max(class_count)
        for i in range(class_total):
            if mmax == class_count[i]:
                predict.append(i)
                break
    return np.array(predict)




