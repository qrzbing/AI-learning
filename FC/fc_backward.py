# -*- coding:utf-8 -*-
# 反向传播层

import tensorflow as tf
import os
# 引入 mnist 数据集
from tensorflow.examples.tutorials.mnist import input_data
# 引入数据集生成层
import fc_generate
# 引入前向传播层
import fc_forward

# 一次喂入的数据量
BATCH_SIZE =           200
# 基础学习率
LEARNING_RATE_BASE =   0.1
# 学习率衰减率
LEARNING_RATE_DECAY =  0.99
# 正则化系数
REGULARIZER =          0.0001
# 轮数
STEPS =                50000
# 滑动平均衰减率
MOVING_AVERAGE_DECAY = 0.99
# 手动给出总样本数
train_num_examples =   60000

# 模型的保存路径
MODEL_SAVE_PATH = "./model/"
# 模型的保存文件名
MODEL_NAME = "mnist_model"

# 反向传播函数
def backward():
    print("[+] define x, y_")
    # 由于不知道要喂几组，所以 shape 的第一个参数为 None
    # 输入特征，每个标签有  INPUT_NODE 个元素
    x  = tf.placeholder(tf.float32, [None,  fc_forward.INPUT_NODE])
    # 标准答案，每个标签有 OUTPUT_NODE 个元素
    y_ = tf.placeholder(tf.float32, [None, fc_forward.OUTPUT_NODE])
    # 前向训练过程
    print("[+] start forward")
    y = fc_forward.forward(x, REGULARIZER)
    # 定义训练轮数计数器
    global_step = tf.Variable(0, trainable = False)
    print("[+] define loss function")
    # 损失函数
    # 让模型的输出经过softmax函数，以获得输出分类的概率分布，
    # 再与标准答案对比，求出交叉熵，得到损失函数
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits = y,
        labels = tf.argmax(y_, 1)
    )
    cem = tf.reduce_mean(ce)
    # 调用包含正则化的损失函数 loss
    loss = cem + tf.add_n(tf.get_collection('losses'))
    print("[+] define learning rate")
    # 学习率
    learning_rate = tf.train.exponential_decay(
        # 学习率计数，超参数
        LEARNING_RATE_BASE,
        # 学习轮数计数器
        global_step,
        # 多少轮更新一次学习率，为 训练样本数/一轮喂入数据
        train_num_examples / BATCH_SIZE,
        # 学习率衰减率
        LEARNING_RATE_DECAY,
        # staircase 
        # 为 True 时 global_step / LEARNING_RATE_STEP
        # 取整数，学习率阶梯型衰减
        # 为 False 时则为一条平滑下降的曲线
        staircase = True
    )
    print("[+] define train step")
    # 定义训练过程
    # 使用梯度下降的方法，以减小 loss 值为目标
    train_step = tf.train.GradientDescentOptimizer(
        learning_rate
    ).minimize(
        loss,
        global_step = global_step
    )
    print("[+] define exponential moving average")
    # 滑动平均(影子值)
    ema = tf.train.ExponentialMovingAverage(
        # 衰减率，超参数，一般为比较大的值
        MOVING_AVERAGE_DECAY,
        # 当前轮数
        global_step)
    # ema.apply() 定义为对括号内的参数求滑动平均
    # tf.trainable_variables() 可以自动将待训练的参数汇总成列表
    ema_op = ema.apply(tf.trainable_variables())
    # 在工程应用中，我们常把计算滑动平均和训练过程绑定在一起运行，
    # 他们合成一个训练节点，用下述语句实现：
    with tf.control_dependencies([train_step, ema_op]):
        train_op = tf.no_op(name = 'train')
    print("[+] define saver")
    # 实例化 saver
    saver = tf.train.Saver()
    img_batch, label_batch = fc_generate.get_tfRecord(
        BATCH_SIZE, 
        isTrain = True
    )
    print("[+] define Session, start learning")
    with tf.Session() as sess:
        # 初始化所有变量
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        # 断点重续
        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        print("[+] start coordinator")
        # 线程协调器
        # 开启
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(
            sess = sess,
            coord = coord
        )
        print("[+] start iteration")
        # 迭代 STEPS 轮
        for i in range(STEPS):
            # print("[---+---] ", i)
            # 从训练集中随机抽取 BATCH_SIZE 组数据和标签，把它们喂入神经网络执行训练过程
            # print("label_batch", label_batch)
            xs, ys = sess.run([img_batch, label_batch])
            _, loss_value, step = sess.run(
                [train_op, loss, global_step],
                feed_dict={x: xs, y_: ys}
            )
            # 每一千轮后输出 loss 值
            if i % 1000 == 0:
                print(
                    "[+] After %d training step(s), loss on training batch is %g." \
                     % (step, loss_value)
                )
                # 保存值
                saver.save(
                    sess,
                    os.path.join(
                        MODEL_SAVE_PATH,
                        MODEL_NAME
                    ),
                    global_step = global_step
                )

        # 关闭
        coord.request_stop()
        coord.join(threads)

def main():
    print("[+] start backward")
    backward()

if __name__ == '__main__':
    main()