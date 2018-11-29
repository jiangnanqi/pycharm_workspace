import tensorflow as tf
from tensorflow.python.platform import gfile

with tf.Session() as sess:
    model_filename = './model/combined_model.pd'
    with gfile.FastGFile(model_filename,mode='rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    result = tf.import_graph_def(graph_def,return_elements=['add:0'])
    print(sess.run(result))