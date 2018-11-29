import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time
import numpy as np

import minist_inference_CNN
from Full_link import mnist_train

EVAL_INTERVAL_SECS = 10

def evaluate(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32,
                           shape=[mnist.validation.images.shape[0], minist_inference_CNN.IMAGE_SIZE, minist_inference_CNN.IMAGE_SIZE,
                                  minist_inference_CNN.NUM_CHANNELS], name='x-input')
        y_ = tf.placeholder(tf.float32, shape=[None, minist_inference_CNN.OUTPUT], name='y-input')

        print(mnist.validation.images.shape)

        reshaped_xs1 = np.reshape(mnist.validation.images, (mnist.validation.images.shape[0], minist_inference_CNN.IMAGE_SIZE, minist_inference_CNN.IMAGE_SIZE,minist_inference_CNN.NUM_CHANNELS))
        validation_feed = {x: reshaped_xs1, y_: mnist.validation.labels}
        test_feed = {x: mnist.test.images, y_: mnist.test.labels}

        y = minist_inference_CNN.inference(x,False,None)

        correct_prediction = tf.equal(tf.argmax(y_,1),tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

        variable_averages = tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DEACY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)


        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess,ckpt.model_checkpoint_path)

                    # global_step = ckpt.model_checkpoint_path.split('/')[-1].split ('-')[-1]
                    accuracy_score = sess.run(accuracy,feed_dict=validation_feed)
                    print('After ' + str(0) + ' training step(s) , validation accuracy using average model is ' + str(accuracy_score))
                else:
                    print('No checkpoint file found')
                    return
            time.sleep(EVAL_INTERVAL_SECS)
def main(argv=None):
    mnist = input_data.read_data_sets('../data/',one_hot=True)
    evaluate(mnist)
if __name__ == '__main__':
    tf.app.run()