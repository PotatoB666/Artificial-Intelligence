import tensorflow as tf
import numpy as np
import random

batch_size = 60000

#training data
x_data = np.linspace(-.5,5,2000)[:,np.newaxis]
y_data = tf.sin(x_data)

xs = tf.placeholder('float',[None,1,1])
ys = tf.placeholder('float',[None,1])

weights = {
    'in':tf.Variable(tf.random_normal([1,100],stddev=0.1)),
    'out':tf.Variable(tf.random_normal([100,1],stddev=0.1))
}
biases = {
    'in':tf.Variable(tf.zeros([100])),
    'out':tf.Variable(tf.zeros([1]))
}

def RNN(x,Weights,biases):
    x = tf.reshape(x,[-1,1])
    x_in = tf.matmul(x,Weights['in'])+biases['in']
    x_in = tf.reshape(x_in,[-1,1,1])
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(100,forget_bias=1.,state_is_tuple=True)
    init_state = lstm_cell.zero_state(2000,dtype=tf.float32)
    output,final_state = tf.nn.dynamic_rnn(lstm_cell,x_in,initial_state=init_state,time_major=False)
    outputs = tf.matmul(final_state[1],Weights['out'])+biases['out']
    return outputs

prediction = RNN(xs,weights,biases)

loss = tf.reduce_mean(tf.square(prediction-ys))

train_op = tf.train.AdamOptimizer(1e-4).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(0,300):
        sess.run(train_op,feed_dict={xs:x_data.reshape([-1,1,1]),ys:y_data})
        m_loss = sess.run(loss,feed_dict={xs:x_data,ys:y_data})
        print(m_loss)