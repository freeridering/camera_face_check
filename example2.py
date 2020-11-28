import tensorflow as tf
import numpy as np

# create data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

# tf structure start
Weight = tf.Variable(tf.random_uniform([1], -1, 1))  # 1*1 变量 服从-1到1的平均分布
biases = tf.Variable(tf.zeros([1]))  # 1*1变量，[0]
y = Weight * x_data + biases
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5) # 参数是学习效率
train = optimizer.minimize(loss)
init = tf.global_variables_initializer()
# tf struct end

sess = tf.Session()
sess.run(init) # 激活
for step in range(200):
    sess.run(train)
    if step % 20 == 0:
        print(step,sess.run(Weight),sess.run(biases))

