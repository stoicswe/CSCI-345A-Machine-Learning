import tensorflow as tf
import numpy as np

X = tf.constant([[0.0,0.0,1.0],[0.0,1.0,1.0],[1.0,0.0,1.0],[1.0,1.0,1.0]], name="X")
y = tf.transpose(tf.constant([[0.0, 1.0, 1.0, 0.0]]), name="y")

syn0 = tf.Variable(tf.random_uniform([3,4]) - 1)
syn1 = tf.Variable(tf.random_uniform([4,1]) - 1)

l1 = 1 / (1 + tf.exp(-1*(tf.matmul(X,syn0))))
l2 = 1 / (1 + tf.exp(-1*(tf.matmul(l1,syn1))))

cost = tf.reduce_mean(tf.square(y - l2))

g0, g1 = tf.gradients(cost, [syn0, syn1])

train0 = tf.assign_sub(syn0, g0)
train1 = tf.assign_sub(syn1, g1)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for j in range(6000):
        sess.run([train1, train0])
    pl2 = l2.eval()
    data = X.eval()
    labels = y.eval()

print("X          y          y_pred")
print("----------------------------")
for i in range(len(labels)):
    print(data[i], labels[i], [pl2[i][0]])