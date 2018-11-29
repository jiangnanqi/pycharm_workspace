from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets('data/', one_hot=True)

# mnist数据集相关的数据
input_size = 784
output_size = 10

# 配置神经网络的参数
layer1_node = 500
batch_size = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 300000
MOVING_AVERAGE_DECAY = 0.99


def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
    if avg_class == None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        return tf.matmul(layer1, weights2) + biases2
    else:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1)) + avg_class.average(biases1))
        return tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2)


def train(mnist):
    x = tf.placeholder(tf.float32, shape=(None, input_size), name='x-input')
    y_ = tf.placeholder(tf.float32, shape=(None, output_size), name='y-input')

    weights1 = tf.Variable(tf.truncated_normal([input_size, layer1_node], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[layer1_node]))
    weights2 = tf.Variable(tf.truncated_normal([layer1_node, output_size], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[output_size]))
    y = inference(x, None, weights1, biases1, weights2, biases2)

    global_step = tf.Variable(0, trainable=False)
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    # averages_y = inference(x,variable_averages,weights1,biases1,weights2,biases2)

    cross_entropy_mean = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    regularization = regularizer(weights1) + regularizer(weights2)
    loss = cross_entropy_mean + regularization
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step=global_step,
                                               decay_steps=mnist.train.num_examples / batch_size,
                                               decay_rate=LEARNING_RATE_DECAY)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    train_op = tf.group([train_step, variable_averages_op])

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accruacy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        init_opt = tf.global_variables_initializer()
        sess.run(init_opt)

        validation_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
        test_feed = {x: mnist.test.images, y_: mnist.test.labels}

        for i in range(TRAINING_STEPS):
            if i % 1000 == 0:
                validate_acc = sess.run(accruacy, feed_dict=validation_feed)
                print('After ' + str(i) + ' training step(s) , validation accuracy using average model is ' + str(validate_acc))

            xs, ys = mnist.train.next_batch(batch_size)
            sess.run(train_op, feed_dict={x: xs, y_: ys})
        test_acc = sess.run(accruacy, feed_dict=test_feed)
        print('After ' + str(TRAINING_STEPS) + ' training step(s) , test accuracy using average model is ' + str(
            test_acc))


def main(argv=None):
    mnist = input_data.read_data_sets('data/', one_hot=True)
    train(mnist)


if __name__ == '__main__':
    tf.app.run()
