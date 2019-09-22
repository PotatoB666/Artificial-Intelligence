import tensorflow as tf
import numpy as np
import tensorflow.examples.tutorials.mnist.input_data as input_data

mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

#超参数
nExamples = mnist.train.num_examples
batch_size = 128
nSteps = 28
nInputs = 28
nClassics = 10
nHiddenLayers = 256
nTimes = nExamples // batch_size

x = tf.placeholder(tf.float32,[None,nSteps,nInputs])
y = tf.placeholder(tf.float32,[None,nClassics])

Weights = {
    'in':tf.Variable(tf.random_normal([nInputs,nHiddenLayers],stddev=.1)),
    'out':tf.Variable(tf.random_normal([nHiddenLayers,nClassics],stddev=.1))
}

biases = {
    'in':tf.Variable(tf.constant(.1,shape=[nHiddenLayers,])),
    'out':tf.Variable(tf.constant(.1,shape=[nClassics,]))
}

def RNN(x,weights,biases_n):
    x = tf.reshape(x,[-1,nInputs])
    x_in = tf.matmul(x,weights['in'])+biases_n['in']
    x_in = tf.reshape(x_in,[-1,nSteps,nHiddenLayers])

    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(nHiddenLayers)
    init_state = lstm_cell.zero_state(batch_size,tf.float32)

    output,final_state = tf.nn.dynamic_rnn(lstm_cell,x_in,initial_state=init_state,time_major=False)
    out = tf.matmul(final_state[1],weights['out'])+biases_n['out']
    return out

prediction = RNN(x,Weights,biases)

loss = tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction)
train_op = tf.train.GradientDescentOptimizer(.1).minimize(loss)

correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for rounds in range(500):
        batch_xs,batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape([batch_size,nSteps,nInputs])
        sess.run(train_op,feed_dict={x:batch_xs,y:batch_ys})
        acc = sess.run(accuracy,feed_dict={x:batch_xs,y:batch_ys})
        print(acc)