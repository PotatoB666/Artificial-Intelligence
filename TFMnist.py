import tensorflow as tf
import numpy as np
import tensorflow.examples.tutorials.mnist.input_data as input_data

mnist = input_data.read_data_sets('MNIST_data',one_hot=True)


#批次大小
batch_size = 100
#批次数量
n_batch = mnist.train.num_examples // batch_size

learn_rating = tf.Variable(1e-3,tf.float32)

#参数概要
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean',mean)#平均值
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))#标准差
        tf.summary.scalar('max',tf.reduce_max(var))#最大值
        tf.summary.scalar('min',tf.reduce_min(var))#最小值
        tf.summary.histogram('hitogram',var)#直方图

# 命名空间
with tf.name_scope('input'):
    #placeholder
    x = tf.placeholder(tf.float32,[None,784],name='x-input')
    y = tf.placeholder(tf.float32,[None,10],name='y-input')

#Dropout神经元保存百分比
keep_prob = tf.placeholder(tf.float32)

with tf.name_scope('layer1'):
    #第一层神经元
    with tf.name_scope('Wights'):
        Weights_l1 = tf.Variable(tf.random_normal([784,100]),name='W1')
        variable_summaries(Weights_l1)
    with tf.name_scope('biases'):
        biases_l1 = tf.Variable(tf.zeros([100]),name='b1')
        variable_summaries(biases_l1)
    with tf.name_scope('Wx_plus_b'):
        Wx_plus_b_l1 = tf.matmul(x,Weights_l1)+biases_l1
    with tf.name_scope('Dropout'):
        prediction_l1 = tf.nn.dropout(Wx_plus_b_l1,keep_prob)

with tf.name_scope('layer2'):
    #输出层神经元
    with tf.name_scope('Weights'):
        Weights_l2 = tf.Variable(tf.random_normal([100,10]),name="W2")
    with tf.name_scope('biases'):
        biases_l2 = tf.Variable(tf.zeros([10]),name="b2")
    with tf.name_scope('Wx_plus_b2'):
        Wx_plus_b_l2 = tf.matmul(prediction_l1,Weights_l2)+biases_l2
    with tf.name_scope('prediction2'):
        prediction_l2 = Wx_plus_b_l2

with tf.name_scope('lossFunction'):
    #损失函数
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction_l2))
    tf.summary.scalar('loss',loss)
with tf.name_scope('trainStep'):
    train_step = tf.train.AdagradOptimizer(0.1).minimize(loss)

with tf.name_scope('Accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction_l2,1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        tf.summary.scalar('accuracy', accuracy)

init = tf.global_variables_initializer()

#合并所有的summary
merged = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter('logs/',sess.graph)
    for round in range(31):
        #更新学习率
        sess.run(tf.assign(
            learn_rating,learn_rating*(0.95**round)))
        for _ in range(n_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            summary,_ = sess.run([merged,train_step],feed_dict={x:batch_xs,y:batch_ys,keep_prob:1})
        print(round,sess.run(accuracy,feed_dict={x:batch_xs,y:batch_ys,keep_prob:1}))
        writer.add_summary(summary,round)