import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sess = tf.InteractiveSession()
# 定义单位矩阵：5*5
I_matirx = tf.eye(5)
print(I_matirx.eval())
# 定义变量，初始化为10*10的单位矩阵
X = tf.Variable(tf.eye(10))
X.initializer.run()  # 初始化变量
print(X.eval())  # 计算结果并打印结果
# 定义随机5*10矩阵变量
A = tf.Variable(tf.random_normal([5, 10]))
A.initializer.run()
print(A.eval())


