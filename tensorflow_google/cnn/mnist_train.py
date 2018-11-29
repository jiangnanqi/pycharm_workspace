import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import minist_inference_CNN
import numpy as np

BATCH_SIZE = 100
REGULARAZTION_RATE = 0.0001
MOVING_AVERAGE_DEACY = 0.99

LEARNING_RATE = 0.2
LEARNING_RATE_DECAY = 0.9

TRAINING_STEPS = 6000
MODEL_SAVE_PATH = './model1/'
MODEL_NAME = 'model.ckpt'

def train(mnist):
    x = tf.placeholder(tf.float32,shape=[BATCH_SIZE,minist_inference_CNN.IMAGE_SIZE,minist_inference_CNN.IMAGE_SIZE,minist_inference_CNN.NUM_CHANNELS],name='x-input')
    y_ = tf.placeholder(tf.float32,shape=[None,minist_inference_CNN.OUTPUT],name='y-input')

    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    y = minist_inference_CNN.inference(x,True,regularizer)

    global_step = tf.Variable(0,trainable=False)
    variable_averages = tf.train.ExponentialMovingAverage(decay=MOVING_AVERAGE_DEACY,num_updates=global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    cross_entropy_mean = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1)))
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

    learning_rate = tf.train.exponential_decay(LEARNING_RATE,global_step,mnist.train.num_examples/BATCH_SIZE,decay_rate=LEARNING_RATE_DECAY)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    train_op = tf.group([train_step,variable_averages_op])

    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for i in range(TRAINING_STEPS):
            xs,ys = mnist.train.next_batch(BATCH_SIZE)
            reshaped_xs = np.reshape(xs,(BATCH_SIZE,minist_inference_CNN.IMAGE_SIZE,minist_inference_CNN.IMAGE_SIZE,minist_inference_CNN.NUM_CHANNELS))
            _,loss_value,step = sess.run([train_op,loss,global_step],feed_dict={x:reshaped_xs,y_:ys})

            if i % 50 ==0:
                print('After ' + str(i) + ' training step(s) , batch is ' + str(loss_value))
                saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME))
def main(argv=None):
    mnist = input_data.read_data_sets('../data/',one_hot=True)
    train(mnist)
if __name__ == '__main__':
    tf.app.run()