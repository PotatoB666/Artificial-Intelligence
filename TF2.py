import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#定义数据集
x_data = np.linspace(-.5,.5,200)[:,np.newaxis]
noise = np.random.normal(0,.02,x_data.shape)
y_data = np.square(x_data)+noise

#placeholder
x = tf.placeholder(tf.float32,[None,1])
y = tf.placeholder(tf.float32,[None,1])

#中间层
Weights_l1 = tf.Variable(tf.random_normal([1,10]))
biases_l1 = tf.Variable(tf.zeros([1,10]))
Wx_plus_b_l1 = tf.matmul(x,Weights_l1)+biases_l1
L1 = tf.nn.tanh(Wx_plus_b_l1)

#输出层
Weights_l2 = tf.Variable(tf.random_normal([10,1]))
biases_l2 = tf.Variable(tf.zeros([1,1]))
Wx_plus_b_l2 = tf.matmul(L1,Weights_l2)+biases_l2
L2 = tf.nn.tanh(Wx_plus_b_l2)

#代价函数
loss = tf.reduce_mean(tf.square(L2-y_data))
#优化器
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(4000):
        sess.run(train_step,feed_dict={x:x_data,y:y_data})
        if i%50==0:
            print(sess.run(loss,feed_dict={x:x_data,y:y_data}))

    prediction_value = sess.run(L2, feed_dict={x: x_data})

    plt.figure()
    plt.scatter(x_data, y_data)
    plt.plot(x_data, prediction_value, 'r-', lw=5)
    plt.show()