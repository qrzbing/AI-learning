# -*- coding:utf-8 -*-
'''
File: f:\github\AI-learning\KNN\knn-classify.py
Project: f:\github\AI-learning\KNN
Created Date: Sunday July 22nd 2018
Author: QRZ
-----
Last Modified: Sunday July 22nd 2018 5:55:23 pm
Modified By: QRZ at <qrzbing@foxmail.com>
-----
Copyright (c) 2018 nuaa
'''
import numpy as np
import operator


def classify(inX, dataSet, labels, k, debug=False):
    """
    inx[1,2,3]
    DS=[[1,2,3],[1,2,0]]
    inX: 用于分类的输入向量
    dataSet: 输入的训练样本集
    labels: 标签向量
    k: 选择最近邻居的数目
    注意：labels元素数目和dataSet行数相同；程序使用欧式距离公式.

    预测数据所在分类可在输入下列命令
    kNN.classify0([0,0], group, labels, 3)
    """
    # -----------实现 classify0() 方法的第一种方式-----------
    # 1. 距离计算
    dataSetSize = dataSet.shape[0]  # shape[0]返回dataSet的行数
    if debug is True:
        print("[DEBUG] dataSetSize: ")
        print(dataSetSize)
    # tile生成和训练样本对应的矩阵，并与训练样本求差
    """
    tile: 列-3表示复制的行数， 行-1／2表示对inx的重复的次数

    In [8]: tile(inx, (3, 1))
    Out[8]:
    array([[1, 2, 3],
        [1, 2, 3],
        [1, 2, 3]])

    In [9]: tile(inx, (3, 2))
    Out[9]:
    array([[1, 2, 3, 1, 2, 3],
        [1, 2, 3, 1, 2, 3],
        [1, 2, 3, 1, 2, 3]])
    """
    # 在列向量方向上重复inX共1次(横向)，
    #   行向量方向上重复inX共dataSetSize次(纵向)
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    if debug is True:
        print("[DEBUG] diffMat: ")
        print(diffMat)
    """
    欧氏距离： 点到点之间的距离
       第一行： 同一个点 到 dataSet的第一个点的距离。
       第二行： 同一个点 到 dataSet的第二个点的距离。
       ...
       第N行： 同一个点 到 dataSet的第N个点的距离。

    [[1,2,3],[1,2,3]]-[[1,2,3],[1,2,0]]
    (A1-A2)^2+(B1-B2)^2+(c1-c2)^2
    """
    # 取平方
    sqDiffMat = diffMat ** 2
    if debug is True:
        print("[DEBUG] sqDiffMat: ")
        print(sqDiffMat)
    # 将矩阵的每一行相加, sum()所有元素相加，sum(0)列相加，sum(1)行相加
    sqDistances = sqDiffMat.sum(axis=1)
    if debug is True:
        print("[DEBUG] sqDistances: ")
        print(sqDistances)
    # 开方,计算出距离
    distances = sqDistances ** 0.5
    if debug is True:
        print("[DEBUG] distances:")
        print(distances)
    # 根据距离排序从小到大的排序，返回对应的索引位置
    # argsort() 是将x中的元素从小到大排列，提取其对应的index（索引），然后输出到y。
    # 例如：x=array([3,0,2,1,4,5])
    #       y = x.argsort()
    #       print(y)
    #       print(x[y])
    # print('distances=', distances)
    sortedDistIndicies = distances.argsort()
    if debug is True:
        print("[DEBUG] sortedDistIndicies: ")
        print(sortedDistIndicies)

    # 2. 选择距离最小的k个点
    classCount = {}  # 一个记录类别次数的字典
    for i in range(k):
        # 取出前k个元素的类别,找到该样本的类型
        voteIlabel = labels[sortedDistIndicies[i]]
        # 在字典中将该类型加一
        # 字典的get方法
        # 如：list.get(k,d) 其中 get相当于一条if...else...语句,参数k在字典中，字典将返回list[k];
        #     如果参数k不在字典中则返回参数d,如果K在字典中则返回k对应的value值
        # l = {5:2,3:4}
        # print(l.get(3,0))返回的值是4；
        # Print(l.get(1,0))返回值是0；
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

    # 3. 排序并返回出现最多的那个类型
    # python3中用items()替换python2中的iteritems(), 以列表返回可遍历的(键，值)元组数组。
    # key=operator.itemgetter(1)根据字典的值进行排序
    # key=operator.itemgetter(0)根据字典的键进行排序
    # reverse降序排序字典

    # 例如：dict = {'Name': 'Zara', 'Age': 7}
    # print("Value : %s" %  dict.items())
    # => Value : [('Age', 7), ('Name', 'Zara')])
    sortedClassCount = sorted(
        classCount.items(),
        key=operator.itemgetter(1),
        reverse=True
    )
    if debug is True:
        print("[DEBUG] item, key: ")
        for value1, value2 in sortedClassCount:
            print(
                "\titem: %s, key: %d"
                % (value1, value2)
            )
    # 返回次数最多的类别,即所要分类的类别
    return sortedClassCount[0][0]

    # ------------------------------------------------------------------------------------------------------------------------------------------
    # 实现 classify0() 方法的第二种方式

    # """
    # 1. 计算距离

    # 欧氏距离： 点到点之间的距离
    #    第一行： 同一个点 到 dataSet的第一个点的距离。
    #    第二行： 同一个点 到 dataSet的第二个点的距离。
    #    ...
    #    第N行： 同一个点 到 dataSet的第N个点的距离。

    # [[1,2,3],[1,2,3]]-[[1,2,3],[1,2,0]]
    # (A1-A2)^2+(B1-B2)^2+(c1-c2)^2

    # inx - dataset 使用了numpy broadcasting，
    # 见 https://docs.scipy.org/doc/numpy-1.13.0/user/basics.broadcasting.html
    # np.sum() 函数的使用
    # 见 https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.sum.html
    # """
    #   dist = np.sum((inx - dataset)**2, axis=1)**0.5

    # """
    # 2. k个最近的标签

    # 对距离排序使用numpy中的argsort函数，
    # 见 https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.sort.html#numpy.sort
    # 函数返回的是索引，因此取前k个索引使用[0 : k]
    # 将这k个标签存在列表k_labels中
    # """
    # k_labels = [labels[index] for index in dist.argsort()[0 : k]]
    # """
    # 3. 出现次数最多的标签即为最终类别

    # 使用collections.Counter可以统计各个标签的出现次数，most_common返回出现次数最多的标签tuple，
    # 例如[('lable1', 2)]，因此[0][0]可以取出标签值
    # """
    # label = Counter(k_labels).most_common(1)[0][0]
    # return label
