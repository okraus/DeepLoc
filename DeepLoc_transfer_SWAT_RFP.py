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

import numpy as np
import preprocess_images as procIm
import h5py
import nn_layers
import tensorflow as tf
import copy
import cPickle
import os

if not os.path.exists('./pretrained_models/model.ckpt-9500.meta'):
    raise NameError('please download pretrained model and extract to ./pretrained_models')

# define new DeepLoc network
numClasses = 11
numChannels = 2

is_training = tf.placeholder(tf.bool, [], name='is_training') # for batch norm
inputs = tf.placeholder('float32', shape = [None,60,60,numChannels], name='inputs')  # for batch norm
labels = tf.placeholder('float32', shape = [None,numClasses], name ='labels')
keep_prob = tf.placeholder(tf.float32)

conv1 = nn_layers.conv_layer(inputs, 3, 3, numChannels, 64, 1, 'conv_1', is_training=is_training)
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
fc2_drop = tf.nn.dropout(fc_2,keep_prob)
logits = nn_layers.nn_layer(fc2_drop, 512, numClasses, 'final_layer', act=None, is_training=is_training)
y = tf.nn.softmax(logits, name = 'pred_layer')


# identify layers to load from DeepLoc network trained on Chong et al. 2015 data

variables2restore = []
newVariables = []
for x in tf.global_variables():
    if 'final_layer' in x.name or 'pred_layer' in x.name:
        newVariables.append(x)
    else:
        variables2restore.append(x)


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

if os.path.exists('./datasets'):
    print '\nusing full dataset\n'
    dataBaseDir = './datasets/'
else:
    print '\nusing small snapshot dataset\nplease download full dataset for reasonable performance\n'
    dataBaseDir = './datasets_small/'

trainSetPath = dataBaseDir+'Schuldiner_train_set.hdf5'
testSetPath = dataBaseDir+'Schuldiner_test_set.hdf5'


trainH5 = h5py.File(trainSetPath,'r')
testH5 = h5py.File(testSetPath,'r')

localizationTerms = ['ER','bud','bud_neck', 'cell_periphery', 'cytosol', 'mitochondria',
       'nuclear_periphery', 'nucleus', 'punctate', 'vacuole','vacuole_membrane']


def getSpecificChannels(flatImageData,channels,imageSize=64):
    return np.hstack(([flatImageData[:,c*imageSize**2:(c+1)*imageSize**2] for c in channels]))

def processBatch(curBatch):
    curImages = getSpecificChannels(curBatch['data'],[0,1])
    curLabels = curBatch['Index'][:]

    cropSize = 60
    stretchLow = 0.1 # stretch channels lower percentile
    stretchHigh = 99.9 # stretch channels upper percentile
    imSize = 64
    numChan = 2
    processedBatch=procIm.preProcessImages(curImages,
                               imSize,cropSize,numChan,
                               rescale=False,stretch=True,
                               means=None,stds=None,
                               stretchLow=stretchLow,stretchHigh=stretchHigh,
                               jitter=True,randTransform=True)
    return {'data':processedBatch,'Index':curLabels}

def processBatchTest(curBatch):
    curImages = getSpecificChannels(curBatch['data'],[0,1])
    curLabels = curBatch['Index'][:]

    cropSize = 60
    stretchLow = 0.1 # stretch channels lower percentile
    stretchHigh = 99.9 # stretch channels upper percentile
    imSize = 64
    numChan = 2
    processedBatch=procIm.preProcessTestImages(curImages,
                               imSize,cropSize,numChan,
                               rescale=False,stretch=True,
                               means=None,stds=None,
                               stretchLow=stretchLow,stretchHigh=stretchHigh)
    return {'data':processedBatch,'Index':curLabels}


def getTestPerformance(testBatchAll,y):
    testAllPred = np.zeros_like(testBatchAll['Index'])
    batchSize = 500
    for i in range(len(testBatchAll['Index'])/batchSize+1):
        processedBatch = testBatchAll['data'][(i*batchSize):((i+1)*batchSize)]
        crop_list = np.zeros((len(processedBatch), 5, numClasses))
        for crop in range(5):
            images = processedBatch[:, crop, :, :, :]
            tmp = copy.copy(sess.run(y, feed_dict={inputs: images, is_training: False,keep_prob:1.0}))
            crop_list[:, crop, :] = tmp

        testAllPred[(i*batchSize):((i+1)*batchSize)] = np.mean(crop_list, 1)

    testAllAcc = (np.argmax( testBatchAll['Index'],1)==np.argmax( testAllPred,1)).sum()/float(len(testAllPred))
    return testAllPred, testAllAcc

def generateTrainSets(numberPerClass,localizationTerms,trainHdf5):
    trainSetData = []
    trainSetLabels = []

    for loc in range(len(localizationTerms)):
        curNumberPerClass = numberPerClass
        if numberPerClass > (trainHdf5['Index1'][:,loc]==1).sum():
            curNumberPerClass = (trainHdf5['Index1'][:,loc]==1).sum()
        selectedInds = np.random.choice(np.where(trainHdf5['Index1'][:,loc]==1)[0],
                                        curNumberPerClass,
                                        replace=False)
        selectedInds.sort()
        trainSetData.append(trainHdf5['data1'][selectedInds,:])
        trainSetLabels.append(np.float32(trainHdf5['Index1'][selectedInds,:]))

    trainSetData = np.vstack(trainSetData)
    trainSetLabels = np.vstack(trainSetLabels)

    return {'data':trainSetData,'Index':trainSetLabels}


testBatchAll = processBatchTest({'data':testH5['data1'][:],'Index':np.float32(testH5['Index1'][:])})


starter_learning_rate = 0.003
cross_entropy = loss_logits(logits,labels)
train_acc = accuracy(y,labels)
train_step = tf.train.AdamOptimizer(starter_learning_rate).minimize(cross_entropy)

locNetCkpt = './pretrained_models/model.ckpt-9500'
saver = tf.train.Saver(variables2restore)
sess = tf.Session()

EPOCHS={1:500,3:500,5:500,10:500,25:200,50:150,100:100,250:50,500:50}
miniBatchSize = 128
numBootStraps = 5

testBatch = {'data':testBatchAll['data'][:miniBatchSize,0,:],
             'Index':testBatchAll['Index'][:miniBatchSize,:]}

for numberPerClass in [0,1,3,5,10,25,50,100,250,500]:

    train_accs={}
    test_accs={}
    train_costs={}
    test_costs={}
    testAllPred={}
    testAllAcc={}
    for bootstrap in range(numBootStraps):

        iteration = 0
        train_accs[bootstrap]=[]
        test_accs[bootstrap]=[]
        train_costs[bootstrap]=[]
        test_costs[bootstrap]=[]

        sess.run(tf.global_variables_initializer(),{is_training:True})
        ###############################################################
        saver.restore(sess, locNetCkpt) # comment to train from scratch
        ###############################################################

        if numberPerClass>0:
            origTrainBatch = generateTrainSets(numberPerClass, localizationTerms, trainH5)
            print('epoch\tacc\ttest_acc\tcost\ttest_cost')

            #shuffle train batch
            if len(origTrainBatch['Index'])>miniBatchSize:
                shuffleInd = np.arange(len(origTrainBatch['Index']))
                np.random.shuffle(shuffleInd)
                origTrainBatch['Index'] = origTrainBatch['Index'][shuffleInd]
                origTrainBatch['data'] = origTrainBatch['data'][shuffleInd]

            for epoch in range(EPOCHS[numberPerClass]):
                trainBatch = processBatch(origTrainBatch)
                for i in range(len(origTrainBatch['Index'])/miniBatchSize+1):
                    curImages = trainBatch['data'][(i*miniBatchSize):((i+1)*miniBatchSize)]
                    curLabels = trainBatch['Index'][(i*miniBatchSize):((i+1)*miniBatchSize)]

                    _ , acc, cost = sess.run([train_step, train_acc, cross_entropy],
                                       feed_dict={is_training: True,
                                                  keep_prob:0.5,
                                                  inputs: curImages,
                                                  labels: curLabels})
                    train_accs[bootstrap].append(acc)
                    train_costs[bootstrap].append(cost)

                    iteration+=1
                    if iteration%50==0:

                        test_acc, test_cost = sess.run([train_acc, cross_entropy],
                                                   feed_dict={is_training: False,
                                                              keep_prob:1.0,
                                                              inputs: testBatch['data'],
                                                              labels: testBatch['Index']})
                        test_accs[bootstrap].append(test_acc)
                        test_costs[bootstrap].append(test_cost)
                        print iteration,'\t',acc,'\t',test_acc,'\t',cost,'\t',test_cost

            #get all test pred
        testAllPred[bootstrap],testAllAcc[bootstrap] = getTestPerformance(testBatchAll,y)

        print('n_samples\tn_bag\taccuracy')
        print '\t',numberPerClass,'\t',bootstrap,'\t',testAllAcc[bootstrap]

    if not os.path.exists('./logs/transfer_SWAT_RFP'):
        os.makedirs('./logs/transfer_SWAT_RFP')

    f=open('./logs/transfer_SWAT_RFP/number_training_per_class_'+str(numberPerClass)+'.pkl','wb')

    output_dict = {'train_accs':train_accs,'test_accs':test_accs,
                   'train_costs':train_costs,'test_costs':test_costs,
                   'testAllPred':testAllPred,'testAllAcc':testAllAcc}
    cPickle.dump(output_dict,f)
    f.close()
    saver_final = tf.train.Saver(tf.global_variables())
    saver_final.save(sess, './logs/transfer_SWAT_RFP/number_training_per_class_'+str(numberPerClass)+'.ckpt')
    print "results save in './logs/transfer_SWAT_RFP/number_training_per_class_'" + str(numberPerClass)