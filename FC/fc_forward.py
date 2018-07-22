# -*- coding:utf-8 -*-
# 前向传播层

import tensorflow as tf

INPUT_NODE  = 784  # 输入节点为 784 ,即图片像素大小
OUTPUT_NODE =  10  # 输出节点为 784
LAYER1_NODE = 500  # 第一层节点为 500

# shape 为传入的形状，regularizer 为正则化权重
def get_weight(shape, regularizer):
    # 定义权重 w 为正态分布，标准差为 0.1 
    w = tf.Variable(tf.truncated_normal(shape, stddev = 0.1))
    # 正则化缓解过拟合
    if regularizer != None:
        # 这里使用了 l2 正则化
        tf.add_to_collection(
            'losses',
            tf.contrib.layers.l2_regularizer(
                regularizer
            )(w)
        )
    # 返回权重值
    return w

# shape 为传入的形状
def get_bias(shape):
    # 偏置常数为全 0 数组
    b = tf.Variable(tf.zeros(shape))
    # 返回偏置常数值
    return b

# 前向传播函数
def forward(x, regularizer):
    # 第一层神经网络
    w1 = get_weight([INPUT_NODE,  LAYER1_NODE], regularizer)
    b1 = get_bias([LAYER1_NODE])
    # 此函数为实现矩阵乘法加上偏置b1过非线性函数relu()的输出
    # 非线性函数 relu: f(x) = MAX(x, 0)
    y1 = tf.nn.relu(tf.matmul(x, w1) + b1)
    # 第二层神经网络
    w2 = get_weight([LAYER1_NODE, OUTPUT_NODE], regularizer)
    b2 = get_bias([OUTPUT_NODE])
    # 矩阵乘法加偏置
    y = tf.matmul(y1, w2) + b2
    return y
