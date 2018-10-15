import tensorflow as tf
import time
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

import pandas as pd
file = "titanic.csv"

df = pd.read_csv(file)
df['Gender'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

# In[9]:

input_gender = []
input_Pclass = []
output_y = []


input_gender = df.Gender.tolist()
input_Pclass = df.Pclass.tolist()
  
input_x =[]

for x,i in zip(input_gender,input_Pclass):
    input_x.append([x,i])
#print(input_x)

output = df.Survived.tolist()
output_y = [[i] for i in output]

# This program initializes a neural network with four
# hidden layers. This adds power in how much the
# network can handle, and therefor can handle more
# complex datasets, such as linear regression.

X = tf.placeholder(tf.float32, shape=[887,2], name = 'X') #the second value is arbitruary
Y = tf.placeholder(tf.float32, shape=[887,1], name = 'Y')

W = tf.Variable(tf.truncated_normal([2,2]), name = "W")
w = tf.Variable(tf.truncated_normal([2,1]), name = "w")

W1 = tf.Variable(tf.truncated_normal([2,2]), name = "W1")
w1 = tf.Variable(tf.truncated_normal([2,1]), name = "w1")

c = tf.Variable(tf.zeros([887,2]), name = "c")
b = tf.Variable(tf.zeros([887,1]), name = "b")

c1 = tf.Variable(tf.zeros([887,2]), name = "c1")

with tf.name_scope("hidden_layer") as scope:
    h = tf.nn.relu(tf.add(tf.matmul(X, W),c))

with tf.name_scope("output") as scope:
    y_estimated = tf.sigmoid(tf.add(tf.matmul(h, W1),c1))

with tf.name_scope("loss") as scope:
    loss = tf.reduce_mean(tf.squared_difference(y_estimated, Y)) 

with tf.name_scope("train") as scope:
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

#INPUT_XOR = [[0,0],[0,1],[1,0],[1,1]]
#OUTPUT_XOR = [[0],[1],[1],[0]]

init = tf.global_variables_initializer()
sess = tf.Session()

writer = tf.summary.FileWriter("./logs/xor_logs", sess.graph)

sess.run(init)

t_start = time.clock()
for epoch in range(100001):
    sess.run(train_step, feed_dict={X: input_x, Y: output_y})
    if epoch % 10000 == 0:
        print("_"*80)
        print('Epoch: ', epoch)
        """print('   y_estimated: ')
        for element in sess.run(y_estimated, feed_dict={X: input_x, Y: output_y}):
            print('    ',element)
        print('   W: ')
        for element in sess.run(W4):
            print('    ',element)
        print('   c: ')
        for element in sess.run(c4):
            print('    ',element)
        print('   w: ')
        for element in sess.run(w4):
            print('    ',element)
        print('   b ')
        for element in sess.run(b):
            print('    ',element)"""
        print('   loss: ', sess.run(loss, feed_dict={X: input_x, Y: output_y}))
t_end = time.clock()
print("_"*80)
print('Elapsed time ', t_end - t_start)