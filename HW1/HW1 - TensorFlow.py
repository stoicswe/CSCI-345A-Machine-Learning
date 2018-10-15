import tensorflow as tf

cur_x = tf.get_variable("cur_x", initializer=tf.constant([6.0]))
gamma = 0.01
precision = 0.00001
previous_precision = 10.0 
max_iters = 10000
iters = 0

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	while (previous_precision > precision) & (iters < max_iters):
		#initialize the derivitave
		df = 2*cur_x-4
		prev_x = cur_x
		#cur_x = tf.subtract(cur_x, gamma * df)
		cur_x = cur_x - gamma * df
		#df = tf.subtract(tf.multiply(4.0,tf.pow(cur_x,3)), tf.multiply(9.0,tf.pow(cur_x,2)))
		[ _ ,cur_x_val, prev_x_val]  = sess.run([df,cur_x, prev_x])
		previous_precision = abs(cur_x_val - prev_x_val)
		print(previous_precision)
		iters+=1

print("The local minimum occurs at", cur_x_val)