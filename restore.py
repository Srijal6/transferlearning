from __future__ import print_function
from utils import gaussian_kernel_matrix
from mmd1 import gaus
# # Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
from mmd3 import mmd_loss, maximum_mean_discrepancy
from main_model import multilayer_perceptrons
import tensorflow as tf

# Parameters
learning_rate = 0.001
batch_size = 100
display_step = 1
model_path = "/tmp/lenet/model.ckpt"

# Network Parameters
n_hidden_1 = 256 # 1st layer number of features
n_hidden_2 = 256
n_hidden_3= 256
n_hidden_4= 256# 2nd layer number of features
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])


def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)

    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.relu(layer_3)
    # Output layer with linear activation
    out_layer1 = tf.matmul(layer_3, weights['out1']) + biases['out1']
    layer_3 = tf.nn.relu(out_layer1)
    out_layer2 = tf.matmul(layer_3, weights['out2']) + biases['out2']
    return out_layer2


weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1]), trainable=False),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]), trainable=False),
    'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3]), trainable=False),
    'out1': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4])),
    'out2': tf.Variable(tf.random_normal([n_hidden_4, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1]), trainable=False),
    'b2': tf.Variable(tf.random_normal([n_hidden_2]), trainable=False),
    'b3': tf.Variable(tf.random_normal([n_hidden_3]), trainable=False),
    'out1': tf.Variable(tf.random_normal([n_hidden_4])),
    'out2': tf.Variable(tf.random_normal([n_classes]))
}


soarce = (weights['h3'], biases['b3'])
target = (weights['out2'], biases['out2'])
weight = (weights['out2'])

# sigma =
maximum_mean_discrepancy(soarce, target, 5)
gaussian_kernel_matrix(soarce, target, 1e-6)

mmd_loss = mmd_loss(soarce, target, weight)
# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
gamma = 1
# mmd_loss = gaus(soarce, y)

mmd = (cost + gamma * mmd_loss)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(mmd)


init = tf.global_variables_initializer()

saver = tf.train.Saver()

# Running a new session
print("Starting session...")
with tf.Session() as sess:
    # Initialize variables
    sess.run(init)

    # Restore model weights from previously saved model
    saver.restore(sess, model_path)
    print("Model restored from file: %s" % model_path)

    # Resume training
    for epoch in range(5):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                          y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=", \
                "{:.9f}".format(avg_cost))
        print (sess.run(biases['b1']))
    print("Second Optimization Finished!")

    print(sess.run(soarce))

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval(
        {x: mnist.test.images, y: mnist.test.labels}))
