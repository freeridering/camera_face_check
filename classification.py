from __future__ import print_function
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
mnist = input_data.read_data_sets(
    "C:/Users/mentong/anaconda3/envs/py35cv331tf190/Lib/site-packages/tensorflow/mnist_data", one_hot=True)


def add_layer(inputs, in_size, out_size, activation_function=None):
    with tf.name_scope('layer'):
        with tf.name_scope('Weight'):
            Weight = tf.Variable(tf.random_normal([in_size, out_size]), name='W')  # 随机变量初始值一般优于全0
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')  # 推荐biases初始值不为0
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs, Weight) + biases
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        return outputs


def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result


# 创建数据
x_data = np.linspace(-1, 1, 300, dtype=np.float32)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise
# 创建占位符
xs = tf.placeholder(tf.float32, [None, 784], name='x_input')  # [None, 1]表示没有行数要求，但只能有一列
ys = tf.placeholder(tf.float32, [None, 10], name='y_input')
prediction = add_layer(xs, 784, 10, activation_function=tf.nn.softmax)  # softmax一般用于分类
# loss
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
# 初始化变量
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
for i in range(10001):
    # training
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
    if i % 50 == 0:
        print(compute_accuracy(mnist.test.images, mnist.test.labels))
