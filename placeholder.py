import tensorflow as tf

input1 = tf.placeholder(tf.float32)  # 给定数据类型，一般默认float32
input2 = tf.placeholder(tf.float32)  # 给定数据类型，一般默认float32
# input2 = tf.placeholder(tf.float32,[2,2])  # 给定数据类型，一般默认float32 ，规定结构2*2

output = tf.multiply(input1, input2)
with tf.Session() as sess:
    # feed_dict为字典格式，键为placeholder，
    print(sess.run(output, feed_dict={input1: [7.], input2: [2.]}))
