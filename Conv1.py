import tensorflow as tf
import numpy as np
import tensorflow.examples.tutorials.mnist as input_data

mnist = input_data.input_data.read_data_sets('MNIST_data',one_hot=True)
x_data = mnist.train.images
y_data = mnist.train.labels

#批次大小
batch_size = 2000
n_batch = mnist.train.num_examples // batch_size

#placeholder
x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])
keep_prob = tf.placeholder(tf.float32)

#权值初始化函数
def Weight_vaiable(shape):
    return tf.Variable(tf.truncated_normal(shape,stddev=0.1))
#偏置初始化函数
def biases_variable(shape):
    return tf.Variable(tf.zeros(shape=shape))
#卷积神经层
def conv(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
#池化层
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

#数据转化
x_input = tf.reshape(x,[-1,28,28,1])

with tf.name_scope('W_conv1'):
    #初始化第一层卷积层
    Weights_conv1 = Weight_vaiable([5,5,1,32])
with tf.name_scope('b_conv1'):
    biases_conv1 = biases_variable([32])

with tf.name_scope('W_conv2'):
#初始化第二层卷积层
    Weights_conv2 = Weight_vaiable([3,3,32,64])
with tf.name_scope('b_conv2'):
    biases_conv2 = biases_variable([64])

with tf.name_scope('W_fc1'):
#第一层全连接神经网络初始化
    Weights_fc1 = Weight_vaiable([7*7*64,100])
with tf.name_scope('b_fc1'):
    biases_fc1 = biases_variable([100])

with tf.name_scope('W_fc2'):
    #第二次全连接神经网络初始化
    Weights_fc2 = Weight_vaiable([100,10])
with tf.name_scope('b_fc2'):
    biases_fc2 = biases_variable([10])

#########################################################################

#第一层卷积层与池化层
h_conv1 = tf.nn.relu(conv(x_input,Weights_conv1)+biases_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#第二层卷积层与池化层
h_conv2 = tf.nn.relu(conv(h_pool1,Weights_conv2)+biases_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#扁平化
h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])

#第一层全连接神经网络
Wx_plus_b1 = tf.matmul(h_pool2_flat,Weights_fc1)+biases_fc1
h_fc1 = tf.nn.dropout(tf.nn.relu(Wx_plus_b1),keep_prob)

#第二层全连接神经网络
Wx_plus_b2 = tf.matmul(h_fc1,Weights_fc2)+biases_fc2
h_fc2 = tf.nn.softmax(Wx_plus_b2)
prediction = tf.nn.dropout(h_fc2,keep_prob)

with tf.name_scope('loss'):
    #loss函数
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
    tf.summary.scalar('Loss',loss)
train_step = tf.train.GradientDescentOptimizer(0.6).minimize(loss)

with tf.name_scope('accuracy'):
    #准确率计算
    correct_prediction = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    tf.summary.scalar('accuracy',accuracy)

merged = tf.summary.merge_all()
#saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('logs/',sess.graph)
    for rounds in range(21):
        for _ in range(n_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            summary,_ = sess.run([merged,train_step],feed_dict={x:batch_xs,y:batch_ys,keep_prob:1.})
        writer.add_summary(summary,rounds)
        acc1 = sess.run(accuracy,feed_dict={x:batch_xs,y:batch_ys,keep_prob:1.})
        acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels,keep_prob:1.})
        print(rounds,acc1,acc)


    #saver.save(sess,'logs/MNIST_recognazation.ckpt')