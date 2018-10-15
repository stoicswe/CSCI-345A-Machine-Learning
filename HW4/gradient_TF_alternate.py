import tensorflow as tf

gamma = 0.01
precision = 0.00001
previous_precision = 10.0 
max_iters = 100000
iters = 0

x = tf.Variable(0.)
y = x**4 - 3 * x**3 + 2
df = tf.gradients([y],[x])

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	x0 = 6.0
	while (iters < max_iters):
		x1 = x0 - gamma * sess.run(df, {x:x0})[0]
		x0 = x1
		iters+=1
	cur_x_val = x1

print("The local minimum occurs at", cur_x_val)