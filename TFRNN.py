import tensorflow as tf
import random
import tensorflow.examples.tutorials.mnist.input_data as input

mnist = input.read_data_sets('MNIST_data',one_hot=True)

n_steps = 28
n_inputs = 28
batch_size = 128
n_hidden_layers = 128
n_classies = 10
lr = 1e-4

x = tf.placeholder('float',[None,n_steps,n_inputs])
y = tf.placeholder('float',[None,n_classies])

Weights = {
    'in':tf.Variable(tf.random_normal([n_inputs,n_hidden_layers],stddev=0.1)),
    'out':tf.Variable(tf.random_normal([n_hidden_layers,n_classies],stddev=0.1))
}

biases = {
    'in':tf.Variable(tf.constant(0.1,shape=[n_hidden_layers,])),
    'out':tf.Variable(tf.constant(0.1,shape=[n_classies,]))
}

def RNN(X,Weights,biases):
    X = tf.reshape(X,[-1,n_inputs])
    X_in = tf.matmul(X,Weights['in'])+biases['in']
    X_in = tf.reshape(X_in,[-1,n_steps,n_hidden_layers])

    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_layers,forget_bias=1.0,state_is_tuple=True)
    init_state = lstm_cell.zero_state(batch_size,dtype=tf.float32)

    output,final_state = tf.nn.dynamic_rnn(lstm_cell,X_in,initial_state=init_state,time_major=False)

    outputs = tf.matmul(final_state[1],Weights['out'])+biases['out']
    return outputs

prediction = RNN(x,Weights,biases)

loss = tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction)
train_op = tf.train.AdamOptimizer(lr).minimize(loss)


correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sum = 0
    for round in range(3000):
        batch_xs,batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape([batch_size,n_steps,n_inputs])
        sess.run(train_op,feed_dict={x:batch_xs,y:batch_ys})
        acc = sess.run(accuracy,feed_dict={x:batch_xs,y:batch_ys})
        sum += batch_size
        print('accuracy:'+str(acc),' - samples:'+str(sum))
