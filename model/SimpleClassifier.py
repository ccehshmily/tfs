from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Import data
from ..datasource.HistoryDataUtil import HistoryDataUtil as dUtil

import tensorflow as tf
import math

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', '/tmp/data/', 'Directory for storing data')

dailyData = dUtil.extractDailyData('tfs/datasource/sampledata/sampleGOOG.txt')
dataSet = dUtil.generateTFData(dailyData, N=10, L=1, dateInfoWanted=False)
featureLen = len(dataSet[0][0][0])

classifiedClasses = 10
hidden1_units = int(featureLen / 2)
hidden2_units = int(featureLen / 2)

print(featureLen)

sess = tf.InteractiveSession()

# Create the model
x = tf.placeholder(tf.float32, [None, featureLen])
#W = tf.Variable(tf.truncated_normal([featureLen, classifiedClasses], stddev=1.0 / math.sqrt(float(featureLen))))
#W = tf.Variable(tf.zeros([featureLen, classifiedClasses]))
#b = tf.Variable(tf.zeros([classifiedClasses]))
#y = tf.nn.softmax(tf.matmul(x, W) + b)

# Hidden 1
with tf.name_scope('hidden1'):
    #weights = tf.Variable(tf.truncated_normal([featureLen, hidden1_units], stddev=1.0 / math.sqrt(float(featureLen))), name='weights')
    weights = tf.Variable(tf.zeros([featureLen, hidden1_units]), name='weights')
    biases = tf.Variable(tf.zeros([hidden1_units]), name='biases')
    hidden1 = tf.nn.tanh(tf.matmul(x, weights) + biases)
# Hidden 2
with tf.name_scope('hidden2'):
    #weights = tf.Variable(tf.truncated_normal([hidden1_units, hidden2_units], stddev=1.0 / math.sqrt(float(hidden1_units))), name='weights')
    weights = tf.Variable(tf.zeros([hidden1_units, hidden2_units]), name='weights')
    biases = tf.Variable(tf.zeros([hidden2_units]), name='biases')
    hidden2 = tf.nn.tanh(tf.matmul(hidden1, weights) + biases)
# Linear
with tf.name_scope('softmax_linear'):
    #weights = tf.Variable(tf.truncated_normal([hidden2_units, classifiedClasses], stddev=1.0 / math.sqrt(float(hidden2_units))), name='weights')
    weights = tf.Variable(tf.zeros([hidden2_units, classifiedClasses]), name='weights')
    biases = tf.Variable(tf.zeros([classifiedClasses]), name='biases')
    logits = tf.nn.tanh(tf.matmul(hidden2, weights) + biases)

y = tf.nn.softmax(logits)

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, classifiedClasses])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.3).minimize(cross_entropy)

with tf.Session() as sess:
    # Train
    tf.initialize_all_variables().run()
    for i in range(200):
        (batch_xs, batch_ys) = dataSet[0]
        train_step.run({x: batch_xs, y_: batch_ys})
        if (i - 1)%50 == 0:
            print("finished training round: " + str(i))
            # Test trained model
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            print("train: ", accuracy.eval({x: dataSet[0][0], y_: dataSet[0][1]}))
            print("test: ", accuracy.eval({x: dataSet[1][0], y_: dataSet[1][1]}))

    print(sess.run([y], feed_dict={x: dataSet[0][0]})[0][0:20])
    print(sess.run([y], feed_dict={x: dataSet[1][0]})[0][0:20])
