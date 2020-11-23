import tensorflow as tf

matrix1 = tf.constant([[3, 3]])
matrix2 = tf.constant([[2], [2]])
result = tf.matmul(matrix1, matrix2)

# 执行
# 方法一
sess = tf.Session()
print(sess.run(result))
sess.close()
# 方法二
with tf.Session() as  sess:
    print(sess.run(result))
