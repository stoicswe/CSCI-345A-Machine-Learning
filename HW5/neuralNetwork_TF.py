import tensorflow as tf
import numpy as np

X = [[0.0,0.0,1.0],[0.0,1.0,1.0],[1.0,0.0,1.0],[1.0,1.0,1.0]]
y = [[0.0, 1.0, 1.0, 0.0]]
data = np.array(X)
labels = np.array(y).T
syn0 = tf.random_uniform([3,4]) - 1 # 4 is arbitrary, 3 is for inputs
syn1 = tf.random_uniform([4,1]) - 1 # 4 is arbitrary, 1 is for outputs

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for j in range(600):
        l1 = 1 / (1 + tf.exp(-1*(tf.matmul(X,syn0))))
        l2 = 1 / (1 + tf.exp(-1*(tf.matmul(l1,syn1))))
        l2_delta = (y - l2) * (l2 * (1-l2))
        l1_delta = tf.matmul(l2_delta, (tf.transpose(syn1) * (l1 * (1 - 1*l1))))
        syn1 += tf.matmul(tf.transpose(l1), l2_delta)
        syn0 += tf.matmul(tf.transpose(X), l1_delta)
    pl2 = l2.eval()
pl2 = np.array(pl2)[0]
print("X    y    y_pred")
print("----------------")
for i in range(len(labels)):
    print(data[i], labels[i], [pl2[i]])