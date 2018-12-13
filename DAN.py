from __future__ import print_function

import tensorflow as tf

import os
from tensorflow import math

from my_model import cnn
import data_loader


os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Training settings
batch_size = 32
epochs = 200
lr = 0.01
momentum = 0.9
no_cuda = False
seed = 8
log_interval = 10
l2_decay = 5e-4
root_path = "./dataset/"
source_name = "amazon"
target_name = "webcam"
no_of_batches= 20

cuda = not no_cuda and tf.test.is_gpu_available()

tf.random.set_random_seed(seed)
if cuda:
    tf.device(seed)  # tf.matmul(seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

source_loader = data_loader.load_training(root_path, source_name, batch_size, kwargs)
target_train_loader = data_loader.load_training(root_path, target_name, batch_size, kwargs)
target_test_loader = data_loader.load_testing(root_path, target_name, kwargs)

len_source_dataset = len(source_loader.dataset)
len_target_dataset = len(target_test_loader.dataset)
len_source_loader = len(source_loader)
len_target_loader = len(target_train_loader)


def load_pretrain(model):
    # saver = tf.train.import_meta_graph('my_test_model-1000.meta')
    # https://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
    # have to add my trained model
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph('my_test_model-1000.meta')
        new_saver.restore(sess, tf.train.latest_checkpoint('./'))
    for k, v in model_dict.items():
        if not "cls_fc" in k:
            model_dict[k] = pretrained_dict[k[k.find(".") + 1:]]
    model.load_state_dict(model_dict)
    return model


def train(epoch, model):
    # LEARNING_RATE = lr / tf.math((1 + 10 * (epoch - 1) / epochs), 0.75)
    # print('learning rate{: .4f}'.format(LEARNING_RATE))


    optimizer = tf.train.AdamOptimizer(0.01,beta1=0.9, beta2=0.9, epsilon=l2_decay)

    iter_source = iter(source_loader)
    iter_target = iter(target_train_loader)
    num_iter = len_source_loader
    for i in range(1, num_iter):
        data_source, label_source = iter_source.get_next()
        data_target, _ = iter_target.get_next()
        if i % len_target_loader == 0:
            iter_target = iter(target_train_loader)
        if cuda:
            data_source, label_source = data_source.cuda(), label_source.cuda()  # tf.matmul(data_source)
            data_target = data_target.cuda()  # tf.matmul(data_target)
        data_source, label_source = tf.Variable(data_source), tf.Variable(label_source)
        data_target = tf.Variable(data_target)

        label_source_pred, loss_mmd = model(data_source, data_target)
        loss_cls = tf.losses.sparse_softmax_cross_entropy(label_source_pred, label_source)
        gamma = 2 / (1 + math.exp(-10 * (epoch) / epochs)) - 1
        loss = loss_cls + gamma * loss_mmd

        if i % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tsoft_Loss: {:.6f}\tmmd_Loss: {:.6f}'.format(
                epoch, i * len(data_source), len_source_dataset,
                       100. * i / len_source_loader, loss.data[0], loss_cls.data[0], loss_mmd.data[0]))
    train_op = optimizer.minimize(loss)

    with tf.Session() as sess:
        for i in range(num_batches):
            _, loss_val = sess.run([train_op, loss])

def test(model):
    tf.estimator.Estimator(model)  # is it correct
    test_loss = 0
    correct = 0

    for data, target in target_test_loader:
        if cuda:
            data, target = tf.matmul(data), tf.matmul(target)  # data.device
        data, target = tf.Variable(data ), tf.Variable(target)
        s_output, t_output = model(data, data)
        test_loss +=tf.losses.sparse_softmax_cross_entropy(data,target)
      #  pred= tf.nn.sparse_softmax_cross_entropy_with_logits(s_output), target, size_average=False).data[
       # 0]  # sum up batch loss

        pred = tf.argmax(s_output,1) # get the index of the max log-probability

        correct+= sess.run(pred,feed_dict={data,target})

    test_loss /= len_target_dataset
    print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        target_name, test_loss, correct, len_target_dataset,
        100. * correct / len_target_dataset))
    return correct


if __name__ == '__main__':
    model = cnn(num_classes=31)
    correct = 0
    print(model)
    if cuda:
        model.device()  # tf.matmul(model)
    model = load_pretrain(model)
    for epoch in range(1, epochs + 1):
        train(epoch, model)
        t_correct = test(model)
        if t_correct > correct:
            correct = t_correct
        print('source: {} to target: {} max correct: {} max accuracy{: .2f}%\n'.format(
            source_name, target_name, correct, 100. * correct / len_target_dataset))
