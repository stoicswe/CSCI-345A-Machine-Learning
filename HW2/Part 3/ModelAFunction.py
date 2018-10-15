import tensorflow as tf
import time
import numpy as np

X = tf.placeholder(tf.float32, shape=[1000,1], name = 'X')
Y = tf.placeholder(tf.float32, shape=[1000,1], name = 'Y')
w = tf.Variable(tf.truncated_normal([1,1000]), name = "w")

b = tf.Variable(tf.zeros([1000,1]), name = "b")

with tf.name_scope("output") as scope:
    y_estimated = tf.sigmoid(tf.add(tf.matmul(X,w),b))
    print(y_estimated)

with tf.name_scope("loss") as scope:
    loss = tf.reduce_mean(tf.squared_difference(y_estimated, Y))

with tf.name_scope("train") as scope:
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

x = [[xi] for xi in list(np.linspace(-10,10,1000))]
noise = list(np.random.uniform(-1,1,len(x)))
y = [[3 * x[i][0] + 1 + noise[i]] for i in range(len(x))]

init = tf.global_variables_initializer()
sess = tf.Session()

writer = tf.summary.FileWriter("./logs/xor_logs", sess.graph)

sess.run(init)

t_start = time.clock()
for epoch in range(100001):
    sess.run(train_step, feed_dict={X: x, Y: y})
    if epoch % 1000 == 0:
        print("_"*80)
        print('Epoch: ', epoch)
        print('   w mean:')
        print("   {0}".format(np.mean(sess.run(w))))
        print('   b mean:')
        print("   {0}".format(np.mean(sess.run(b))))
        print('   loss: ', (sess.run(loss, feed_dict={X: x, Y: y})) / 1000)
t_end = time.clock()
print("_"*80)
print('Elapsed time ', t_end - t_start)