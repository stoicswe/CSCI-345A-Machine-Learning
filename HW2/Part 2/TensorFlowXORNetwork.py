#THIS WAS XOR, BUT NOW OR PROBLEM HAS BEEN SOLVED!!!
#Nathan Bunch


# Activation RELU + sigmoid for binary classification output + MSE loss function
import tensorflow as tf
import time
import numpy as np

X = tf.placeholder(tf.float32, shape=[4,2], name = 'X')
Y = tf.placeholder(tf.float32, shape=[4,1], name = 'Y')

W = tf.Variable(tf.truncated_normal([2,2]), name = "W")
w = tf.Variable(tf.truncated_normal([2,1]), name = "w")

#c = tf.Variable(tf.zeros([4,2]), name = "c")
b = tf.Variable(tf.zeros([4,1]), name = "b")

#with tf.name_scope("hidden_layer") as scope:
#   h = tf.nn.relu(tf.add(tf.matmul(X, W),c))
#    print(h)

with tf.name_scope("output") as scope:
    #y_estimated = tf.sigmoid(tf.add(tf.matmul(h,w),b))
    y_estimated = tf.sigmoid(tf.add(tf.matmul(X,w),b))
    print(y_estimated)

with tf.name_scope("loss") as scope:
    loss = tf.reduce_mean(tf.squared_difference(y_estimated, Y)) 

with tf.name_scope("train") as scope:
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

#INPUT_XOR = [[0,0],[0,1],[1,0],[1,1]]
#OUTPUT_XOR = [[0],[1],[1],[0]]

INPUT_OR = [[0,0],[0,1],[1,0],[1,1]]
OUTPUT_OR = [[0],[1],[1],[0]]

init = tf.global_variables_initializer()
sess = tf.Session()

writer = tf.summary.FileWriter("./logs/xor_logs", sess.graph)

sess.run(init)

t_start = time.clock()
for epoch in range(100001):
    sess.run(train_step, feed_dict={X: INPUT_OR, Y: OUTPUT_OR})
    if epoch % 1000 == 0:
        print("_"*80)
        print('Epoch: ', epoch)
        #print('   y_estimated: ')
        #for element in sess.run(y_estimated, feed_dict={X: INPUT_OR, Y: OUTPUT_OR}):
        #    print('    ',element)
        #print('   W: ')
        #for element in sess.run(W):
        #    print('    ',element)
        
        #print('   c: ')
        #for element in sess.run(c):
        # #    print('    ',element)
        
        #print('   w: ')
        #for element in sess.run(w):
        #    print('    ',element)
        #print('   b ')
        #for element in sess.run(b):
        #    print('    ',element)
        print('   loss: ', sess.run(loss, feed_dict={X: INPUT_OR, Y: OUTPUT_OR}))
t_end = time.clock()
print("_"*80)
print('Elapsed time ', t_end - t_start)