import tensorflow as tf

INPUT_BODE = 784
OUTPUT = 10
IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10

CONV1_SIZE = 5
CONV1_DEEP = 32

CONV2_SIZE = 5
CONV2_DEEP = 64

FC_SIZE = 512

def get_weight_variable(shape,regularizer):
    weights = tf.get_variable('weights',shape=shape,initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer != None:
        tf.add_to_collection('losses',regularizer(weights))
    return weights

def inference(input_tensor,train,regularizer):
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable('weight',shape=[CONV1_SIZE,CONV1_SIZE,NUM_CHANNELS,CONV1_DEEP],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_bias = tf.get_variable('bias',shape=[CONV1_DEEP],initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input_tensor,conv1_weights,strides=[1,1,1,1],padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1,conv1_bias))

    with tf.variable_scope('layer2-pooling'):
        pool1 = tf.nn.max_pool(relu1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    with tf.variable_scope('layer3-conv2'):
        weigths = tf.get_variable('weight',shape=[CONV2_SIZE,CONV2_SIZE,CONV1_DEEP,CONV2_DEEP],initializer=tf.truncated_normal_initializer(stddev=0.1))
        bias = tf.get_variable("bias",shape=[CONV2_DEEP],initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1,filter=weigths,strides=[1,1,1,1],padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2,bias=bias))

    with tf.variable_scope('layer4-pooling2'):
        pool2 = tf.nn.max_pool(relu2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    pool2_shape = pool2.get_shape().as_list()
    print(pool2_shape)
    nodes = pool2_shape[1]*pool2_shape[2]*pool2_shape[3]
    print(nodes)
    reshaped = tf.reshape(pool2,shape=[pool2_shape[0],nodes])

    with tf.variable_scope('layer5_fc1'):
        weigths = get_weight_variable(shape=[nodes,FC_SIZE],regularizer=regularizer)
        fc1_biases = tf.get_variable('bias',shape=[FC_SIZE],initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(reshaped,weigths)+fc1_biases)
        if train:
            fc1 = tf.nn.dropout(fc1,0.5)

    with tf.variable_scope('layer6-fc2'):
        weigths = get_weight_variable(shape=[FC_SIZE,OUTPUT],regularizer=regularizer)
        fc2_bias = tf.get_variable('bias',shape=[OUTPUT],initializer=tf.constant_initializer(0.0))
        logit = tf.matmul(fc1,weigths)+fc2_bias
    return logit