import tensorflow as tf
import random
import numpy as np


def next_batch(batch_size):
    A=[]
    C=[]
    for i in range(batch_size):
        a=random.randint(0,4000)
        b=random.randint(0,4000)
        o=random.randint(0,2)
        A.append([a,o,b])
        if o==0:
            C.append(a+b)
        elif o==1:
            C.append(a-b)
        elif o==2:
            C.append(a*b)
    return A,C
#超参数
n_hidden_layers = 1
n_step = 3
n_input = 1
n_batch_size = 1
n_classics = 1

x = tf.placeholder(tf.float32,[None,n_step,n_input])
y = tf.placeholder(tf.float32,[None,n_classics])

Weights = {
    'in':tf.Variable(tf.random_normal([n_input,n_hidden_layers],stddev=.1)),
    'out':tf.Variable(tf.random_normal([n_hidden_layers,n_classics],stddev=.1))
}

biases = {
    'in':tf.Variable(tf.constant(.1,tf.float32,[n_hidden_layers,])),
    'out':tf.Variable(tf.constant(.1,tf.float32,[n_classics,]))
}

def RNN(xI,weights,biases_n):
    xI = tf.reshape(xI,[-1,n_input])
    x_in = tf.matmul(xI,weights['in'])+biases_n['in']
    x_in = tf.reshape(x_in,[-1,n_step,n_hidden_layers])

    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_layers)
    init_state = lstm_cell.zero_state(n_batch_size,dtype=tf.float32)

    output,final_state = tf.nn.dynamic_rnn(
        lstm_cell,
        x_in,
        time_major=False,
        initial_state=init_state
    )
    out = tf.matmul(final_state[1],weights['out'])+biases_n['out']
    return out

prediction = RNN(x,Weights,biases)

loss = tf.reduce_mean(tf.square(y-prediction))
accuracy = 1/tf.reduce_mean(tf.square(y-prediction))
tf.summary.scalar('loss',loss)
train_op = tf.train.AdamOptimizer(1e-5).minimize(loss)

merged = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('logs/')
    for rounds in range(20000):
        xs,ys = next_batch(n_batch_size)
        xs = np.reshape(xs,[n_batch_size,n_step,n_input])
        ys = np.reshape(ys,[n_batch_size,1])
        sess.run([merged,train_op],feed_dict={x:xs,y:ys})
        acc = sess.run(accuracy,feed_dict={x:xs,y:ys})
        print(rounds+1,acc)