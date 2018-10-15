import tensorflow as tf

#========================================================================
print("Doing this in constants:")
sess = tf.Session()
a = tf.constant(2, dtype=tf.int32)
b = tf.constant(4, dtype=tf.int32)
c = tf.constant(6, dtype=tf.int32)
d = a + b
e = d + c
print("{0} + {1} = {2}".format(2,4,sess.run(d)))
print("{0} + {1} = {2}".format(6,6,sess.run(e)))
#========================================================================
print("Doing this in variables:")
sess = tf.Session()
a = tf.get_variable("number1", dtype=tf.int32, initializer=tf.constant(2))
b = tf.get_variable("number2", dtype=tf.int32, initializer=tf.constant(4))
c = tf.get_variable("number3", dtype=tf.int32, initializer=tf.constant(6))
sess.run(tf.global_variables_initializer())
d = a + b
e = d + c
print("{0} + {1} = {2}".format(2,4,sess.run(d)))
print("{0} + {1} = {2}".format(6,6,sess.run(e)))
#=========================================================================
print("Doing this with placeholders:")
sess = tf.Session()
a = tf.placeholder(tf.int32)
b = tf.placeholder(tf.int32)
c = tf.placeholder(tf.int32)
d = a + b
e = d + c
print("{0} + {1} = {2}".format(2,4,sess.run(d, feed_dict={a: 2, b: 4})))
print("{0} + {1} = {2}".format(6,6,sess.run(e, feed_dict={d: 6, c: 6})))