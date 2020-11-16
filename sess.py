import tensorflow as tf
h = tf.constant("hell,TF")
sess=tf.Session()
print(sess.run(h))