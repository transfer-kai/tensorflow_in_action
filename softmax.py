# -*- coding: utf-8 -*-
# tensorflow实现softmax对mnist数据集进行分类
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)
sess=tf.InteractiveSession()
#x为输入数据：784个特征
x=tf.placeholder(tf.float32,[None,784])
#W为权重，[784,10],784个特征，10类;b为位移，10维度，因为label在one-hot编码后为10维
W=tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([10]))
y=tf.nn.softmax(tf.matmul(x,W)+b)
y_=tf.placeholder(tf.float32,[None,10])
cross_entropy=tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices=[1]))
train_step=tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
tf.global_variables_initializer().run()
for i in range(1000):
    batch_xs,batch_ys=mnist.train.next_batch(100)
    train_step.run({x:batch_xs,y_:batch_ys})
correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
print("accuray=",accuracy.eval({x:mnist.test.images,y_:mnist.test.labels}))