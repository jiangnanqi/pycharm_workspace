import tensorflow as tf

INPUT_BODE = 784
OUTPUT = 10
LAYER1_DOME = 500

def get_weight_variable(shape,regularizer):
    weights = tf.get_variable('weights',shape=shape,initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer != None:
        tf.add_to_collection('losses',regularizer(weights))
    return weights

def inference(input_tensor,regularizer):
    with tf.variable_scope('layer1'):
        weights = get_weight_variable([INPUT_BODE,LAYER1_DOME],regularizer)
        biases = tf.get_variable('biases',[LAYER1_DOME],initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor,weights)+biases)

    with tf.variable_scope('layer2'):
        weights = get_weight_variable([LAYER1_DOME,OUTPUT],regularizer)
        biases = tf.get_variable('biases',shape=[OUTPUT],initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1,weights)+biases
    return layer2