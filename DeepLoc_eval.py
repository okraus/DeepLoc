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
import cellDataClass as dataClass # NO QUEUE
import preprocess_images as procIm # NO QUEUE
import numpy as np
import copy
import glob
import cPickle
import os

import argparse

parser = argparse.ArgumentParser(description='Evaluate DeepLoc model on Chong et al., 2015 data')
parser.add_argument("-logdir", action="store", dest="logdir", help="directory to store results")
args = parser.parse_args()
print args.logdir

checkpoint_dir = args.logdir

def DeepLocModel(input_images, is_training):

    conv1 = nn_layers.conv_layer(input_images, 3, 3, 2, 64, 1, 'conv_1', is_training=is_training)
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
    fc_2 = nn_layers.nn_layer(fc_1, 512, 512, 'fc_2', act=tf.nn.relu, is_training=is_training)
    #y = nn_layers.nn_layer(fc_2, 512, 19, 'final_layer', act=tf.nn.softmax, is_training=is_training)
    logit = nn_layers.nn_layer(fc_2, 512, 19, 'final_layer', act=None, is_training=is_training)

    return logit

def loss(predicted_y,labeled_y):
    with tf.name_scope('cross_entropy'):
        diff = labeled_y * tf.log(tf.clip_by_value(predicted_y,1e-16,1.0))
        with tf.name_scope('total'):
            cross_entropy = -tf.reduce_mean(diff)
        tf.scalar_summary('cross entropy', cross_entropy)

    return cross_entropy

def loss_logits(logits,labeled_y):
    with tf.name_scope('cross_entropy'):
        logistic_losses = tf.nn.softmax_cross_entropy_with_logits(logits, labeled_y, name='sigmoid_cross_entropy')
        cross_entropy = tf.reduce_mean(logistic_losses)
        tf.scalar_summary('cross entropy', cross_entropy)

    return cross_entropy


def accuracy(predicted_y,labeled_y):
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(predicted_y, 1), tf.argmax(labeled_y, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.scalar_summary('accuracy', accuracy)

    return accuracy


def loss_numpy(y_pred, y_lab):
    cross_entropy = -np.mean(y_lab * np.log(np.clip(y_pred, 1e-16, 1.0)))
    return cross_entropy


def accuracy_numpy(y_pred, y_lab):
    accuracy = np.mean(np.argmax(y_pred, 1) == np.argmax(y_lab, 1))
    return accuracy


def eval( checkpoint_path):
    print('\n\n', 'evaluating', '\n\n')
    # initialize tf session
    sess = tf.Session()

    global_step = tf.Variable(0, trainable=False)

    ######################
    # DATASET PARAMETERS #
    ######################

    if os.path.exists('./datasets'):
        print '\nusing full dataset\n'
        dataBaseDir = './datasets/'
    else:
        print '\nusing small snapshot dataset\nplease download full dataset for reasonable performance\n'
        dataBaseDir = './datasets_small/'

    trainHdf5 = dataBaseDir+'Chong_train_set.hdf5'
    validHdf5 = dataBaseDir+'Chong_valid_set.hdf5'

    cropSize = 60
    batchSize = 128
    stretchLow = 0.1 # stretch channels lower percentile
    stretchHigh = 99.9 # stretch channels upper percentile

    imSize = 64
    numClasses = 19
    numChan = 2
    loadedDataSets = {}
    loadedDataSets['train'] = dataClass.Data(trainHdf5,['data','Index'],batchSize)
    loadedDataSets['valid'] = dataClass.Data(validHdf5,['data','Index'],batchSize)


    ### define model
    is_training = tf.placeholder(tf.bool, [], name='is_training')  # for batch norm
    inputs = tf.placeholder('float32', shape=[None, 60, 60, 2], name='inputs')  # for batch norm

    logits = DeepLocModel(inputs, is_training)
    predicted_y = tf.nn.softmax(logits, name='softmax')

    sess.run(tf.global_variables_initializer(),{is_training:False})
    saver = tf.train.Saver(tf.global_variables())

    checkpoint_files = glob.glob(checkpoint_path + '/*ckpt-[0-9]*.meta')
    checkpoint_files = [str.split(x, '.meta')[0] for x in checkpoint_files]
    results = {'cost':{'train':[],'valid':[]},
               'acc':{'train':[],'valid':[]},
               'steps':[]}
    print checkpoint_files



    for j, checkpoint_file in enumerate(checkpoint_files):
        saver.restore(sess, checkpoint_file)

        global_step = checkpoint_file.split('/')[-1].split('-')[-1]
        # test loop
        # start training and test queue's

        accList = {'train':[],'valid':[]}
        lossList = {'train': [], 'valid': []}

        for dataSet in ['train','valid']:
            data = loadedDataSets[dataSet]

            numberDataPoints = data.stopInd - data.startInd


            for i in range(2):
            #for i in range(numberDataPoints / data.batchSize):
                crop_list = np.zeros((data.batchSize, 5, numClasses))
                batch = data.getBatch()
                processedBatch=procIm.preProcessTestImages(batch['data'],
                                           imSize,cropSize,numChan,
                                           rescale=False,stretch=True,
                                           means=None,stds=None,
                                           stretchLow=stretchLow,stretchHigh=stretchHigh)
                for crop in range(5):
                    images = processedBatch[:, crop, :, :, :]
                    tmp = copy.copy(sess.run([predicted_y], feed_dict={inputs: images, is_training: False}))
                    crop_list[:, crop, :] = tmp[0]

                mean_crops = np.mean(crop_list, 1)
                curAcc = accuracy_numpy(mean_crops,batch['Index'])
                curCost = loss_numpy(mean_crops,batch['Index'])
                accList[dataSet].append(curAcc)
                lossList[dataSet].append(curCost)

        print('total ' +dataSet+' ' + str(global_step), np.mean(accList[dataSet]),np.mean(lossList[dataSet]))
        results['acc']['train'].append(np.mean(accList['train']))
        results['acc']['valid'].append(np.mean(accList['valid']))
        results['cost']['train'].append(np.mean(lossList['train']))
        results['cost']['valid'].append(np.mean(lossList['valid']))
        results['steps'].append(global_step)

    with open(checkpoint_dir + '/test_acc_deploy_results.pkl', 'wb') as f:
        cPickle.dump(results, f)


def main(_):

    eval(checkpoint_dir)


if __name__ == '__main__':
    tf.app.run()
