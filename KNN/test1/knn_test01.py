# -*- coding:utf-8 -*-
'''
File: f:\github\AI-learning\KNN\test1\knn_test01.py
Project: f:\github\AI-learning\KNN\test1
Created Date: Sunday July 22nd 2018
Author: QRZ
-----
Last Modified: Sunday July 22nd 2018 6:26:06 pm
Modified By: QRZ at <qrzbing@foxmail.com>
-----
Copyright (c) 2018 nuaa
'''

import knn_generate
import knn_classify
import getopt
import sys
import numpy as np
from matplotlib import pyplot as plt


def help(flag):
    if flag is False:
        print("[-] Error args!")
        print("You can input \"python knn_test01.py -h\" for help")
    print("[+] python knn_test01.py -d <flag>")
    print("[+] python knn_test01.py --debug <flag>")


def print_label(label):
    if label == knn_generate.romance_label:
        print("爱情片")
    elif label == knn_generate.action_label:
        print("动作片")


if __name__ == '__main__':
    argv = sys.argv[1:]
    flag = False
    try:
        opts, args = getopt.getopt(argv, "hd:", ["debug="])
    except getopt.GetoptError:
        help(False)
        exit(2)
    for opt, arg in opts:
        if opt == '-h':
            help(True)
        elif opt in ('-d', '--debug'):
            if arg == 'True':
                flag = True
            elif arg == 'False':
                flag = False
            else:
                help(False)
                exit(2)
        elif opt is not None:
            help(False)
    # 创建数据集
    group, labels = knn_generate.createDataSet()
    # 测试集
    test = np.array([[101, 20]])
    # TODO: 作图
    romance_xy = (labels == knn_generate.romance_label)
    # print(group[romance_xy][:, 0])
    action_xy = (labels == knn_generate.action_label)
    plt.plot(
        group[romance_xy][:, 0], group[romance_xy][:, 1], 'ro',
        group[action_xy][:, 0], group[action_xy][:, 1], 'bs',
        test[:, 0], test[:, 1], 'g^'
    )
    plt.show()
    # plt.plot(group[romance_x], )
    # kNN分类
    test_class = knn_classify.classify(test, group, labels, 3, debug=flag)
    # 打印分类结果
    print_label(test_class)
