import tensorflow as tf

x = tf.Variable(0.)
y = 4.0 * tf.pow(x, 3) - 9.0 * tf.square(x)
#f(x) = x**2 and df(x) = 2*x
z = tf.gradients([y], [x])
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for value in range(-20, 20):
		result = sess.run([z], {x:value})  #I show here that a tf.Variable can take feed values
		print(result)
	