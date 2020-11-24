from __future__ import print_function
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def add_layer(inputs, in_size, out_size, activation_function=None):
    with tf.name_scope('layer'):
        with tf.name_scope('Weight'):
            Weight = tf.Variable(tf.random_normal([in_size, out_size]), name='W')  # 随机变量初始值一般优于全0
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1,name='b')  # 推荐biases初始值不为0
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs, Weight) + biases
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        return outputs


# 创建数据
x_data = np.linspace(-1, 1, 300, dtype=np.float32)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise
# 创建占位符
with tf.name_scope('input'):  # 构建可视化的框架
    ys = tf.placeholder(tf.float32, [None, 1], name='x_input')  # [None, 1]表示没有行数要求，但只能有一列
    xs = tf.placeholder(tf.float32, [None, 1], name='y_input')
# 输入层 1个神经元 x_data
# 隐藏层 10个神经元 自定，激活函数
# 输出层 1个神经元 y_data
# 输入层
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
# 隐藏层,假定没有激活函数
prediction = add_layer(l1, 10, 1, activation_function=None)
# reduce_sum为求和，reduce_mean为求平均值
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
# 以一定的学习效率减小误差
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)  # 学习效率通常小于1
# 初始化变量

sess = tf.Session()
writer = tf.summary.FileWriter('logs/',sess.graph)
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()
sess.run(init)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(x_data, y_data)
plt.ion()
plt.show(block=False)

for i in range(1001):
    # training
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
        # to see the step improvement
        # print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        prediction_value = sess.run(prediction, feed_dict={xs: x_data})
        lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
        plt.pause(0.1)
