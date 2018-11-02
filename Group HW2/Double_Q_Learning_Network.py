import gym
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v0')

tf.reset_default_graph()

#These lines establish the feed-forward part of the network used to choose actions
# =========================================================
inputs1 = tf.placeholder(shape=[1,16],dtype=tf.float32)
W1 = tf.Variable(tf.random_uniform([16,4],0,0.01))
Qout1 = tf.matmul(inputs1,W1)
# =========================================================
inputs2 = tf.placeholder(shape=[1,16],dtype=tf.float32)
W2 = tf.Variable(tf.random_uniform([16,4],0,0.01))
Qout2 = tf.matmul(inputs2,W2)
# =========================================================
predict = tf.argmax(Qout1 + Qout2,1)


#Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
nextQ1 = tf.placeholder(shape=[1,4],dtype=tf.float32)
nextQ2 = tf.placeholder(shape=[1,4],dtype=tf.float32)
loss1 = tf.reduce_sum(tf.square(nextQ1 - Qout1))
loss2 = tf.reduce_sum(tf.square(nextQ2 - Qout2))
trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
updateModel1 = trainer.minimize(loss1)
updateModel2 = trainer.minimize(loss2)

init = tf.initialize_all_variables()

# Set learning parameters
y = .99
e = 0.1
num_episodes = 2000
#create lists to contain total rewards and steps per episode
jList = []
rList = []
with tf.Session() as sess:
    sess.run(init)
    for i in range(num_episodes):
        #Reset environment and get first new observation
        s = env.reset()
        rAll = 0
        d = False
        j = 0
        #The Q-Network
        while j < 99:
            j+=1
            #Choose an action by greedily (with e chance of random action) from the Q-network
            a,_,_ = sess.run([predict, Qout1, Qout2],feed_dict={inputs1:np.identity(16)[s:s+1], inputs2:np.identity(16)[s:s+1]})
            allQ1 = sess.run([Qout1],feed_dict={inputs1:np.identity(16)[s:s+1]})
            allQ2 = sess.run([Qout2],feed_dict={inputs2:np.identity(16)[s:s+1]})
            allQ = allQ1 + allQ2

            if np.random.rand(1) < e:
                a[0] = env.action_space.sample()
            #Get new state and reward from environment
            s1,r,d,_ = env.step(a[0])
            #Obtain the Q' values by feeding the new state through our network
            Q1 = sess.run(Qout1,feed_dict={inputs1:np.identity(16)[s1:s1+1]})
            Q2 = sess.run(Qout2,feed_dict={inputs2:np.identity(16)[s1:s1+1]})
            #Obtain maxQ' and set our target value for chosen action.
            maxQ1 = np.max(Q1)
            maxQ2 = np.max(Q2)
            targetQ1 = allQ1
            targetQ2 = allQ2
            print("Targets:")
            print(str(targetQ1))
            print(str(targetQ2))
            print("A:")
            print(str(a[0]))
            print("Maxs:")
            print(str(maxQ1))
            print(str(maxQ2))
            print("TQ0:")
            print(targetQ1[0])
            
            targetQ1[0][0][a[0]] = r + y*(maxQ2)
            targetQ2[0][0][a[0]] = r + y*(maxQ1)
            #Train our network using target and predicted Q values
            x = np.random.rand()
            if x > 0.5:
                _,W1 = sess.run([updateModel1,W1],feed_dict={inputs1:np.identity(16)[s:s+1],inputs2:np.identity(16)[s:s+1],nextQ1:targetQ1})
            else:
                _,W2 = sess.run([updateModel2,W2],feed_dict={inputs1:np.identity(16)[s:s+1],inputs2:np.identity(16)[s:s+1],nextQ2:targetQ2})
            rAll += r
            s = s1
            if d == True:
                #Reduce chance of random action as we train the model.
                e = 1./((i/50) + 10)
                break
        jList.append(j)
        rList.append(rAll)
print("Percent of succesful episodes: " + str(sum(rList)/num_episodes) + "%")