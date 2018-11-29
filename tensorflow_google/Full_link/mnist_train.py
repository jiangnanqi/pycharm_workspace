import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from Full_link import minist_inference

BATCH_SIZE = 100
REGULARAZTION_RATE = 0.0001
MOVING_AVERAGE_DEACY = 0.99
LEARNING_RATE = 0.8
LEARNING_RATE_DECAY = 0.99
TRAINING_STEPS = 100000000
MODEL_SAVE_PATH = './model1/'
MODEL_NAME = 'model.ckpt'

def train(mnist):
    x = tf.placeholder(tf.float32, shape=[None, minist_inference.INPUT_BODE], name='x-input')
    y_ = tf.placeholder(tf.float32, shape=[None, minist_inference.OUTPUT], name='y-input')

    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    y = minist_inference.inference(x, regularizer)

    global_step = tf.Variable(0,trainable=False)
    variable_averages = tf.train.ExponentialMovingAverage(decay=MOVING_AVERAGE_DEACY,num_updates=global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    cross_entropy_mean = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

    learning_rate = tf.train.exponential_decay(LEARNING_RATE,global_step,mnist.train.num_examples/BATCH_SIZE,decay_rate=LEARNING_RATE_DECAY)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    train_op = tf.group([train_step,variable_averages_op])

    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for i in range(TRAINING_STEPS):
            xs,ys = mnist.train.next_batch(BATCH_SIZE)
            _,loss_value = sess.run([train_op,loss],feed_dict={x:xs,y_:ys})

            if i % 1000 ==0:
                print('After ' + str(i) + ' training step(s) , batch is ' + str(loss_value))
                saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME))
def main(argv=None):
    mnist = input_data.read_data_sets('./data/',one_hot=True)
    train(mnist)
if __name__ == '__main__':
    tf.app.run()