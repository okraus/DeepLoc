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
import h5py
import matplotlib
matplotlib.use('Agg') # do not display cells for headless implementations
import matplotlib.pyplot as plt
import numpy as np
from skimage import filters
import mahotas as mh
import argparse
import os


import argparse
parser = argparse.ArgumentParser(description='Visualize DeepLoc model on Chong et al., 2015 data')
parser.add_argument("-l","--logdir",action="store",dest="logdir",help="directory to save models",
                    default='./pretrained_DeepLoc/pretrained_models/model.ckpt-5000')
parser.add_argument("-o", "--output-folder", action="store", dest="outputdir", help="directory to store results",
                    default='./output_figures')
args = parser.parse_args()
print 'log dir:',args.logdir,'out dir:',args.outputdir


locNetCkpt = args.logdir
output_dir = args.outputdir

if not os.path.exists(locNetCkpt+'.meta'):
    raise NameError('please download pretrained model using download_datasets.sh')


#################
# DeepLoc MODEL #
#################

is_training = tf.placeholder(tf.bool, [], name='is_training') # for batch norm
inputs = tf.placeholder('float32', shape = [60,60,2], name='inputs')  # for batch norm
labels = tf.placeholder('float32', shape = [None,19], name ='labels')

input_reshape = tf.reshape(inputs, [1, 60, 60 ,2])
conv1 = nn_layers.conv_layer(input_reshape, 3, 3, 2, 64, 1, 'conv_1', is_training=is_training)
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
lastAct = nn_layers.nn_layer(fc_2, 512, 19, 'final_layer', act=None, is_training=is_training)


# initialize DeepLoc model
sess = tf.Session()
sess.run(tf.global_variables_initializer(),{is_training:False})

# load model checkpoint
saver = tf.train.Saver(tf.global_variables())
saver.restore(sess, locNetCkpt)

# load DeepLoc training data
localizationTerms=['ACTIN', 'BUDNECK', 'BUDTIP', 'CELLPERIPHERY', 'CYTOPLASM',
           'ENDOSOME', 'ER', 'GOLGI', 'MITOCHONDRIA', 'NUCLEARPERIPHERY',
           'NUCLEI', 'NUCLEOLUS', 'PEROXISOME', 'SPINDLE', 'SPINDLEPOLE',
           'VACUOLARMEMBRANE', 'VACUOLE','DEAD','GHOST']

cellDataFile = h5py.File('./datasets/Chong_valid_set.hdf5','r')
labels = cellDataFile['Index1'][:]
images = cellDataFile['data1'][:]
cellDataFile.close()

def getInitImage(inputCell):
    outData = np.zeros((60,60,2))
    outData[:,:,0] = inputCell[:64**2].reshape(64,64)[2:-2,2:-2]
    outData[:,:,1] = inputCell[64**2:].reshape(64,64)[2:-2,2:-2]
    #stretch
    for chan in range(2):
        p_low = np.percentile(outData[:,:,chan],0.1)
        p_high = np.percentile(outData[:,:,chan],99.9)
        outData[:,:,chan] = outData[:,:,chan] - p_low
        outData[:,:,chan] = outData[:,:,chan] / (p_high-p_low)

    return outData

def render_naive_gaussian_blur_l2_clamped(t_obj,
                                          mask,
                                          img0,
                                          iter_n=100,
                                          step=0.1,
                                          sigma=1.,
                                          decay=.1,
                                          clip_val_g=60,
                                          b_every=2):

    t_score = tf.reduce_mean(t_obj) # defining the optimization objective
    t_grad = tf.gradients(t_score, inputs)[0] # automatic differentiation

    img = img0.copy()
    for i in range(iter_n):

        g, score = sess.run([t_grad, t_score], {inputs:img, is_training:False})

        g[:,:,0] /= g[:,:,0].std()+1e-8
        norm = np.abs(g*img)

        img[:,:,0] += g[:,:,0]*step*(1.-i*decay)
        img[:,:,0][norm[:,:,0]<np.percentile(norm[:,:,0],clip_val_g)] = 0

        if i%b_every==0:
            img[:,:,0] = mh.gaussian_filter(img[:,:,0],sigma)
        if mask is not None:
            img[mask==False,0]=0

    return img


# percentile of image area to set to background
useTopPerc={'ACTIN':60,
 'BUDNECK':80,
 'BUDTIP':80,
 'CELLPERIPHERY':60,
 'CYTOPLASM':20,
 'ENDOSOME':60,
 'ER':60,
 'GOLGI':50,
 'MITOCHONDRIA':50,
 'NUCLEARPERIPHERY':70,
 'NUCLEI':60,
 'NUCLEOLUS':70,
 'PEROXISOME':50,
 'SPINDLE':70,
 'SPINDLEPOLE':70,
 'VACUOLARMEMBRANE':50,
 'VACUOLE':50,
 'DEAD':0,
 'GHOST':0}

class2sample_from = 1
# get samples with BUDNECK localization
buddedCells = images[labels[:,class2sample_from]==1]
# pick random budded cell
curInd = np.random.choice(len(buddedCells))
curCell = buddedCells[curInd]



if not os.path.exists(output_dir):
    os.makedirs(output_dir)


plt.figure(figsize=(9,35))
numFigs = len(localizationTerms[:-2])+1

for localizationClass in range(len(localizationTerms[:-2]))[:]:


    # noisy green channel initialization
    print 'generating ',localizationTerms[localizationClass]
    img_noise = np.random.uniform(size=(60,60,2))
    # strech and crop curCell
    initImage=getInitImage(curCell)
    initImage0=initImage.copy()
    # calc mask for pixels outside cell in red channel
    val = filters.threshold_otsu(initImage[:,:,1])
    mask = initImage[:,:,1]>(val*.8)

    # set init green channel to random noise
    initImage[:,:,0]=img_noise[:,:,0]

    if localizationClass==0:
        plt.subplot(numFigs,3,1)
        plt.imshow(initImage[:,:,0],'gray')
        plt.title('green_init')
        plt.axis('off')
        plt.subplot(numFigs,3,2)
        plt.imshow(initImage[:,:,1],'gray')
        plt.title('red_init')
        plt.axis('off')
        plt.subplot(numFigs,3,3)
        plt.imshow(mh.as_rgb(initImage0[:,:,1],initImage0[:,:,0],None))
        plt.title('original')
        plt.axis('off')
        #plt.savefig('./output_figures/'+localizationTerms[class2sample_from]+'_'+str(curInd)+'_init.png')

    generatedCell = render_naive_gaussian_blur_l2_clamped(lastAct[:,localizationClass],mask=mask,
                                          clip_val_g=[useTopPerc[localizationTerms[localizationClass]]],
                                               img0=initImage,iter_n=100,step=.3,sigma=.7,decay=.009)

    plt.subplot(numFigs,3,(localizationClass+1)*3+1)
    plt.imshow(generatedCell[:,:,0],'gray')
    plt.title(localizationTerms[localizationClass])
    plt.axis('off')
    plt.subplot(numFigs,3,(localizationClass+1)*3+2)
    plt.imshow(generatedCell[:,:,1],'gray')
    plt.title(localizationTerms[localizationClass])
    plt.axis('off')
    plt.subplot(numFigs,3,(localizationClass+1)*3+3)
    plt.imshow(mh.as_rgb(generatedCell[:,:,1],generatedCell[:,:,0],None))
    plt.title(localizationTerms[localizationClass])
    plt.axis('off')

#plt.tight_layout()
plt.savefig(output_dir+'/generated_cells.png')
print 'figure saves as: ' + output_dir+'/generated_cells.png'