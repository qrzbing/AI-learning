# -*- coding:utf-8 -*-
'''
File: f:\github\AI-learning\KNN\knn-test01.py
Project: f:\github\AI-learning\KNN
Created Date: Sunday July 22nd 2018
Author: QRZ
-----
Last Modified: Sunday July 22nd 2018 6:26:06 pm
Modified By: QRZ at <qrzbing@foxmail.com>
-----
Copyright (c) 2018 nuaa
'''
# -*- coding: UTF-8 -*-

import knn_generate
import knn_classify
import getopt
import sys


def help(flag):
    if flag is False:
        print("[-] Error args!")
        print("You can input \"python knn_test01.py\" -h for help")
    print("[+] python knn_test01.py -d <flag>")
    print("[+] python knn_test01.py --debug <flag>")


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
    # 创建数据集
    group, labels = knn_generate.createDataSet()
    # 测试集
    test = [101, 20]
    # kNN分类
    test_class = knn_classify.classify(test, group, labels, 3, debug=flag)
    # 打印分类结果
    print(test_class)
