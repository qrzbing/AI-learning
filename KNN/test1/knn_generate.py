# -*- coding:utf-8 -*-
'''
File: f:\github\AI-learning\KNN\test1\knn_generate.py
Project: f:\github\AI-learning\KNN\test1
Created Date: Sunday July 22nd 2018
Author: QRZ
-----
Last Modified: Sunday July 22nd 2018 5:29:39 pm
Modified By: QRZ at <qrzbing@foxmail.com>
-----
Copyright (c) 2018 nuaa
'''

import sys
import getopt
import numpy as np

romance_label = 0
action_label = 1


def createDataSet():
    # input: [101, 20]
    group = np.array(
        [
            [1, 101],
            [5, 89],
            [108, 5],
            [115, 8]
        ]
    )
    labels = np.array([0, 0, 1, 1])
    return group, labels


def knn_generate_help(flag):
    if flag is False:
        print("[-] Error args!")
        print("You can input \"python knn_generate.py\" -h for help")
    print("[+] python knn_generate.py -d <flag>")
    print("[+] python knn_generate.py --debug <flag>")


if __name__ == '__main__':
    argv = sys.argv[1:]
    flag = False
    try:
        opts, args = getopt.getopt(argv, "hd:", ["debug="])
    except getopt.GetoptError:
        knn_generate_help(False)
        exit(2)
    for opt, arg in opts:
        if opt == '-h':
            knn_generate_help(True)
        elif opt in ('-d', '--debug'):
            if arg == 'True':
                flag = True
            elif arg == 'False':
                flag = False
            else:
                knn_generate_help(False)
                exit(2)
    group, labels = createDataSet()
    if flag is True:
        print("[DEBUG] argv: ", argv)
        print("[DEBUG] group: ")
        print(group)
        print("[DEBUG] labels: ")
        print(labels)
    exit(0)
