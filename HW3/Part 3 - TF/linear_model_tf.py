import tensorflow as tf
import time
import numpy as np


class LinearModel:
    def __init__(self, INPUT, OUTPUT):
        self.INPUT = INPUT
        self.YTRUE = OUTPUT
    def run(self):
        X = tf.constant(self.INPUT, shape=np.array(self.INPUT).shape, name = 'X', dtype=tf.float32)
        Y = tf.constant(self.YTRUE, shape=np.array(self.YTRUE).shape, name = 'Y', dtype=tf.float32)
        w = tf.Variable(tf.truncated_normal([np.array(self.YTRUE).shape[1],1]), name = "w")
        b = tf.Variable(tf.zeros(np.array(self.YTRUE).shape), name = "b")

        with tf.name_scope("output") as scope:
            y_estimated = tf.add(tf.matmul(X,w),b)

        with tf.name_scope("loss") as scope:
            #loss = tf.reduce_mean(tf.squared_difference(y_estimated, Y))
            loss = tf.losses.mean_squared_error(labels=Y, predictions=y_estimated)

        with tf.name_scope("train") as scope:
            train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

        init = tf.global_variables_initializer()
        sess = tf.Session()

        sess.run(init)

        t_start = time.clock()
        for epoch in range(100001):
            sess.run(train_step)
            if epoch % 5000 == 0:
                print("_"*80)
                print('Epoch: ', epoch)
                print('   loss: ', sess.run(loss))
        t_end = time.clock()
        print("_"*80)
        print('Elapsed time ', t_end - t_start)
        #print('Output: ', sess.run(y_estimated))
        return sess.run(y_estimated)

x = [[1], [2], [3], [4]]
y_true = [[0], [-1], [-2], [-3]]

linear_model = LinearModel(x, y_true)
y_pred = linear_model.run() #y_pred = wx+b
print(y_pred)