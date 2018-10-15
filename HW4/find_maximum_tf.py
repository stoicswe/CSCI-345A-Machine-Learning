import tensorflow as tf

def f(x):
	return 0.25 * (tf.cos(x[0]-x[1])*tf.cos(x[0]-x[1])+tf.cos(x[0]-x[3])*tf.cos(x[0]-x[3])+tf.cos(x[2]-x[1])*tf.cos(x[2]-x[1])+tf.sin(x[2]-x[3])*tf.sin(x[2]-x[3]))

def da0(x):
	return 0.25 * (-2*tf.cos(x[0]-x[1])*tf.sin(x[0]-x[1])-2*tf.cos(x[0]-x[3])*tf.sin(x[0]-x[3]))

def db0(x):
	return 0.25 * (2*tf.cos(x[0]-x[1])*tf.sin(x[0]-x[1])+2*tf.cos(x[2]-x[1])*tf.sin(x[2]-x[1]))

def da1(x):
	return 0.25 * (-2*tf.cos(x[2]-x[1])*tf.sin(x[2]-x[1])+2*tf.sin(x[2]-x[3])*tf.cos(x[2]-x[3]))

def db1(x):
	return 0.25 * (2*tf.cos(x[0]-x[3])*tf.sin(x[0]-x[3])-2*tf.sin(x[2]-x[3])*tf.cos(x[2]-x[3]))

gamma = 0.01
max_iters = 500
iters = 0

start_val = 0.1

theta_a0 = tf.Variable(name="theta_a0", initial_value=start_val)
theta_a1 = tf.Variable(name="theta_a1", initial_value=start_val + 1.0)
theta_b0 = tf.Variable(name="theta_b0", initial_value=start_val + 0.5)
theta_b1 = tf.Variable(name="theta_b1", initial_value=start_val - 1.0)

thetas = [tf.Variable(0.0), tf.Variable(0.0), tf.Variable(0.0), tf.Variable(0.0)]

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    while (iters < max_iters):
        prev_theta_a0 = theta_a0
        prev_theta_b0 = theta_b0
        prev_theta_a1 = theta_a1
        prev_theta_b1 = theta_b1
        thetas[0] = da0([theta_a0, theta_b0, theta_a1, theta_b1])
        tf.assign_add(theta_a0, gamma * f(thetas))
        thetas[1] = db0([theta_a0, theta_b0, theta_a1, theta_b1])
        tf.assign_add(theta_b0, gamma * f(thetas))
        thetas[2] = da1([theta_a0, theta_b0, theta_a1, theta_b1])
        tf.assign_add(theta_a1, gamma * f(thetas))
        thetas[3] = db1([theta_a0, theta_b0, theta_a1, theta_b1])
        tf.assign_add(theta_b1, gamma * f(thetas))

        if iters % (max_iters/10) == 0:
            print("{0}%".format((iters / max_iters) * 100))
        iters+=1
    print("MAX: {0}".format((f(thetas)).eval()))
    print("Maximums: A0, B0, A1, B1")
    print(thetas[0].eval(), thetas[1].eval(), thetas[2].eval(), thetas[3].eval())