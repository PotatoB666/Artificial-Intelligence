import tensorflow as tf
import numpy as np

x_data = np.random.rand(100)
y_data = x_data*.1+.2

b = tf.Variable(0.)
k = tf.Variable(0.)
y = k * x_data + b

xs = tf.placeholder(tf.float32,x_data.shape)
ys = tf.placeholder(tf.float32,y_data.shape)

loss = tf.reduce_mean(tf.square(y_data-y))
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for step in range(1000):
        sess.run(train,feed_dict={xs:x_data,ys:y_data})
        if step%50==0:
            print(step,sess.run([k,b]))
