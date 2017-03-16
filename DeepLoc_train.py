# Copyright (c) 2017, Oren Kraus All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation and/or
# other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import tensorflow as tf
import nn_layers
import cellDataClass as dataClass
import preprocess_images as procIm
import os


MAX_STEPS = 10000
WORKERS = 6
SAVE_INTERVAL = 500

import argparse
parser = argparse.ArgumentParser(description='Train DeepLoc model on Chong et al., 2015 data')
parser.add_argument("-l","--logdir",action="store",dest="logdir",help="directory to save models",
                    default='./logs')
args = parser.parse_args()
print 'log dir:',args.logdir


checkpoint_dir = args.logdir

def DeepLocModel(input_images,is_training):

    conv1 = nn_layers.conv_layer(input_images, 3, 3, 2, 64, 1, 'conv_1',is_training=is_training)
    conv2 = nn_layers.conv_layer(conv1, 3, 3, 64, 64, 1, 'conv_2', is_training=is_training)
    pool1 = nn_layers.pool2_layer(conv2, 'pool1')
    conv3 = nn_layers.conv_layer(pool1, 3, 3, 64, 128, 1, 'conv_3', is_training=is_training)
    conv4 = nn_layers.conv_layer(conv3, 3, 3, 128, 128, 1, 'conv_4', is_training=is_training)
    pool2 = nn_layers.pool2_layer(conv4, 'pool2')
    conv5 = nn_layers.conv_layer(pool2, 3, 3, 128, 256, 1, 'conv_5', is_training=is_training)
    conv6 = nn_layers.conv_layer(conv5, 3, 3, 256, 256, 1, 'conv_6', is_training=is_training)
    conv7 = nn_layers.conv_layer(conv6, 3, 3, 256, 256, 1, 'conv_7', is_training=is_training)
    conv8 = nn_layers.conv_layer(conv7, 3, 3, 256, 256, 1, 'conv_8', is_training=is_training)
    pool3 = nn_layers.pool2_layer(conv8, 'pool3')
    pool3_flat = tf.reshape(pool3, [-1, 8 * 8 * 256])
    fc_1 = nn_layers.nn_layer(pool3_flat, 8 * 8 * 256, 512, 'fc_1', act=tf.nn.relu, is_training=is_training)
    fc_2 = nn_layers.nn_layer(fc_1, 512, 512, 'fc_2', act=tf.nn.relu,is_training=is_training)
    logit = nn_layers.nn_layer(fc_2, 512, 19, 'final_layer', act=None, is_training=is_training)

    return logit

def loss(predicted_y,labeled_y):
    with tf.name_scope('cross_entropy'):
        diff = labeled_y * tf.log(tf.clip_by_value(predicted_y,1e-16,1.0))
        with tf.name_scope('total'):
            cross_entropy = -tf.reduce_mean(diff)
        tf.summary.scalar('cross entropy', cross_entropy)

    return cross_entropy

def loss_logits(logits,labeled_y):
    with tf.name_scope('cross_entropy'):
        logistic_losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labeled_y, name='sigmoid_cross_entropy')
        cross_entropy = tf.reduce_mean(logistic_losses)
        tf.summary.scalar('cross entropy', cross_entropy)

    return cross_entropy


def accuracy(predicted_y,labeled_y):
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(predicted_y, 1), tf.argmax(labeled_y, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

    return accuracy

def train():

    print('\n\n','training','\n\n')
    sess = tf.Session()

    dequeueSize = 100

    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = 0.1
    decay_step = 25
    decay_rate = 0.96
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step*dequeueSize,
                                               decay_step * dequeueSize, decay_rate, staircase=True)
    tf.summary.scalar('learning_rate', learning_rate)

    ######################
    # DATASET PARAMETERS #
    ######################

    if os.path.exists('./datasets'):
        print '\nusing full dataset\n'
        dataBaseDir = './datasets/'
    else:
        raise NameError('please download datasets using download_datasets.sh')

    trainHdf5 = dataBaseDir+'Chong_train_set.hdf5'
    validHdf5 = dataBaseDir+'Chong_valid_set.hdf5'

    cropSize = 60
    batchSize = 128
    stretchLow = 0.1 # stretch channels lower percentile
    stretchHigh = 99.9 # stretch channels upper percentile

    imSize = 64
    numClasses = 19
    numChan = 2
    data = dataClass.Data(trainHdf5,['data','Index'],batchSize)
    dataTest = dataClass.Data(validHdf5,['data','Index'],batchSize * 2) # larger batch size at test time

    ### define model
    is_training = tf.placeholder(tf.bool, [], name='is_training') # for batch norm
    input = tf.placeholder('float32', shape = [None,cropSize,cropSize,numChan], name='input')  # for batch norm
    labels = tf.placeholder('float32', shape = [None,numClasses], name='labels')  # for batch norm

    logits = DeepLocModel(input, is_training)
    predicted_y = tf.nn.softmax(logits, name='softmax')

    acc = accuracy(predicted_y,labels)
    cross_entropy = loss_logits(logits, labels)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy,global_step=global_step)

    saver = tf.train.Saver(tf.global_variables(),max_to_keep=100)

    # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(checkpoint_dir + '/train',
                                          sess.graph)
    test_writer = tf.summary.FileWriter(checkpoint_dir + '/test',
                                          sess.graph)
    sess.run(tf.global_variables_initializer(),{is_training:True})

    # training loop

    for i in range(MAX_STEPS):

        if i % 50 == 0:  # Record execution stats

            batch = dataTest.getBatch()
            processedBatch=procIm.preProcessImages(batch['data'],
                                       imSize,cropSize,numChan,
                                       rescale=False,stretch=True,
                                       means=None,stds=None,
                                       stretchLow=stretchLow,stretchHigh=stretchHigh,
                                       jitter=True,randTransform=False)

            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            summary, cur_test_acc, cur_test_loss = sess.run([merged, acc, cross_entropy],

                      feed_dict={is_training: False,
                                 input: processedBatch,
                                 labels: batch['Index']},
                        options=run_options,
                        run_metadata=run_metadata)

            train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
            test_writer.add_summary(summary, i)
            print('Adding run metadata for', i)
            print('Valid accuracy at step %s: %s, loss: %s' % (i, cur_test_acc,cur_test_loss))

        batch = data.getBatch()
        processedBatch=procIm.preProcessImages(batch['data'],
                                   imSize,cropSize,numChan,
                                   rescale=False,stretch=True,
                                   means=None,stds=None,
                                   stretchLow=stretchLow,stretchHigh=stretchHigh)

        summary, _ , cur_train_acc, cur_train_loss = sess.run([merged, train_step, acc, cross_entropy],
                                       feed_dict={is_training: True,
                                                  input: processedBatch,
                                                  labels: batch['Index']})
        train_writer.add_summary(summary, i)
        print('Train accuracy at step %s: %s, loss: %s' % (i, cur_train_acc,cur_train_loss))

        if i % SAVE_INTERVAL == 0:
            checkpoint_path = os.path.join(checkpoint_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=i)

    train_writer.close()
    test_writer.close()

def main(_):
    if tf.gfile.Exists(checkpoint_dir):
        tf.gfile.DeleteRecursively(checkpoint_dir)
    tf.gfile.MakeDirs(checkpoint_dir)

    train()


if __name__ == '__main__':
    tf.app.run()
