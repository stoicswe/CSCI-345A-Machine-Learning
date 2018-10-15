import tensorflow as tf

cur_x = tf.get_variable("cur_x", initializer=tf.constant([6.0]))
gamma = 0.01
max_iters = 10000
iters = 0

y = cur_x**4 - 3 * cur_x**3 + 2
df = tf.gradients(y, [cur_x])[0]

training_op = tf.assign(cur_x , cur_x - gamma * df)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	while (iters < max_iters):
		sess.run([training_op])
		iters+=1
	cur_x_val = cur_x.eval()

print("The local minimum occurs at", cur_x_val)