import tensorflow as tf

from keras.datasets import mnist
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
#import numpy as np

x = tf.placeholder(tf.float32, shape=[None, 28, 28], name="inputs")

### Base Model
with tf.variable_scope("base"):
    y_base = tf.placeholder(tf.float32, shape=[None, 8], name="labels")

    # Reshape input
    res = tf.reshape(x, shape=[-1, 28, 28, 1], name="reshape")

    # Conv
    with tf.variable_scope("conv1"):
        c1 = tf.contrib.layers.conv2d(inputs=res,
                                      num_outputs=32,
                                      kernel_size=5,
                                      stride=1,
                                      padding='SAME',
                                      activation_fn=tf.nn.relu,
                                      weights_initializer=tf.initializers.truncated_normal(stddev=0.1),
                                      biases_initializer=tf.initializers.constant(0.1))

    # Pool
    mp1 = tf.nn.max_pool(c1, ksize=[1, 2, 2, 1],
                         strides=[1, 2, 2, 1], padding='SAME', name="maxpool1")

    # Conv
    with tf.variable_scope("conv2"):
        c2 = tf.contrib.layers.conv2d(inputs=mp1,
                                      num_outputs=64,
                                      kernel_size=5,
                                      stride=1,
                                      padding='SAME',
                                      activation_fn=tf.nn.relu,
                                      weights_initializer=tf.initializers.truncated_normal(stddev=0.1),
                                      biases_initializer=tf.initializers.constant(0.1))
    # Pool
    mp2 = tf.nn.max_pool(c2, ksize=[1, 2, 2, 1],
                         strides=[1, 2, 2, 1], padding='SAME', name="maxpool2")

    # Flatten
    flat = tf.layers.flatten(mp2, name="flatten")

    # FC
    with tf.variable_scope("dense1"):
        fc1 = tf.contrib.layers.fully_connected(inputs=flat,
                                                num_outputs=1024,
                                                activation_fn=tf.nn.relu,
                                                weights_initializer=tf.initializers.truncated_normal(stddev=0.1),
                                                biases_initializer=tf.initializers.constant(0.1))

    # Drop
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    dr = tf.nn.dropout(fc1, keep_prob, name="bottleneck")

    # FC
    with tf.variable_scope("logits"):
        fc2_base = tf.contrib.layers.fully_connected(inputs=dr,
                                                     num_outputs=8,
                                                     activation_fn=None,
                                                     weights_initializer=tf.initializers.truncated_normal(stddev=0.1),
                                                     biases_initializer=tf.initializers.constant(0.1))

    #### Model Evaluation

    with tf.variable_scope("cross_entropy"):
        cross_entropy_base = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_base, logits=fc2_base))

    base_vars = [var for var in tf.trainable_variables() if \
                 var.name.startswith("base")]

    with tf.variable_scope("Adam"):
        train_step_base = tf.train.AdamOptimizer(1e-4).minimize(
            cross_entropy_base, var_list=base_vars)

    with tf.variable_scope("correct_prediction"):
        correct_prediction_base = tf.equal(tf.argmax(fc2_base, 1),
                                           tf.argmax(y_base, 1))

    with tf.variable_scope("accuracy"):
        accuracy_base = tf.reduce_mean(tf.cast(correct_prediction_base, tf.float32))
        accuracy_base_summary = tf.summary.scalar("base_accuracy", accuracy_base)

### Transfer Fork
with tf.variable_scope("transfer"):
    y_trans = tf.placeholder(tf.float32, shape=[None, 2], name="labels")

    # FC
    with tf.variable_scope("logits"):
        fc2_trans = tf.contrib.layers.fully_connected(inputs=dr,
                                                      num_outputs=2,
                                                      activation_fn=None,
                                                      weights_initializer=tf.initializers.truncated_normal(stddev=0.1),
                                                      biases_initializer=tf.initializers.constant(0.1))

    #### Model Evaluation
    with tf.variable_scope("cross_entropy"):
        cross_entropy_trans = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_trans, logits=fc2_trans))

    trans_vars = [var for var in tf.trainable_variables() if \
                  var.name.startswith("transfer")]

    with tf.variable_scope("Adam"):
        train_step_trans = tf.train.AdamOptimizer(1e-4).minimize(
            cross_entropy_trans, var_list=trans_vars)

    with tf.variable_scope("correct_prediction"):
        correct_prediction_trans = tf.equal(tf.argmax(fc2_trans, 1),
                                            tf.argmax(y_trans, 1))

    with tf.variable_scope("accuracy"):
        accuracy_trans = tf.reduce_mean(
            tf.cast(correct_prediction_trans, tf.float32))
        accuracy_trans_summary = tf.summary.scalar("transfer_accuracy", accuracy_trans)

        variable_summaries_to_merge = []
        for v in tf.trainable_variables():
            tmp = tf.summary.histogram(v.name, v)
            variable_summaries_to_merge.append(tmp)

(x_train, y_train), (x_test, y_test) = mnist.load_data()


def getDigitsFromDataset(digits, x, y):
    """
    Given an iterable of digits,
    return the portion of x and y that contain
    those digits.
    """
    mask = np.zeros_like(y, dtype=bool)
    for digit in digits:
        mask += (y == digit)

    x = x[mask]
    y = y[mask]

    return x, y


def get_batches(x, y, batch_size=50, num_classes=8):
    '''
    Given a set of features x, and a set of labels y,
    return a generator yields shuffled batch_size tuples of x and y.

    Note that this function truncates x and y, if necessary,
    so that it only returns full batches.
    '''
    ## Shuffle x and y on the same index permutation
    permute = np.random.permutation(range(len(x)))
    x = x[permute]
    y = y[permute]

    ## Truncate x and y so that no batch is empty
    num_batches = len(x) // batch_size
    x, y = x[:num_batches * batch_size], y[:num_batches * batch_size]

    # One-hot encode the labels:
    y = np.eye(num_classes)[y]

    for i in range(0, len(x), batch_size):
        yield x[i:i + batch_size], y[i:i + batch_size]

def main():
    x_train_base, y_train_base = getDigitsFromDataset(x_train, y_train)
    x_test_base, y_test_base = getDigitsFromDataset(x_test, y_test)
    batch_size = 50
    epochs = 1

    # Define a time-stamped directory in which to keep the TensorBoard data for this run
    summaries_dir = "./summaries/" + datetime.now().strftime("%Y%m%d%H%M%S")

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        merged_variables = tf.summary.merge(variable_summaries_to_merge)
        base_writer = tf.summary.FileWriter(summaries_dir + '/base', sess.graph)

        i = 1
        for e in range(epochs):
            for x_, y_ in get_batches(x_train_base, y_train_base, batch_size):
                if i % 50 == 0:
                    train_accuracy, summary, acc_sum = sess.run(
                        [accuracy_base, merged_variables, accuracy_base_summary],
                        feed_dict={x: x_,
                                   y_base: y_,
                                   keep_prob: 1.0})
                    print('step %d, base training accuracy %g' % (i, train_accuracy))
                    base_writer.add_summary(summary, i)
                    base_writer.add_summary(acc_sum, i)

                train_step_base.run(
                    feed_dict={x: x_,
                               y_base: y_,
                               keep_prob: 0.5})
                i += 1

        y_test_base_oh = np.eye(8)[y_test_base]
        print('base test accuracy %g' % accuracy_base.eval(
            feed_dict={
                x: x_test_base,
                y_base: y_test_base_oh,
                keep_prob: 1.0}))

        saver.save(sess, "checkpoints/base.ckpt")
