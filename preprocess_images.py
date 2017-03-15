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

def preProcessImages(batchData,imSize,cropSize,channels,rescale=True,stretch=False,
                     means=None,stds=None,stretchLow=None,stretchHigh=None,jitter=True,randTransform=True):
    
    if rescale:
        batchData = rescaleBatch(batchData,means,stds,imSize,channels)
    if stretch:
        batchData = stretchBatch(batchData, stretchLow, stretchHigh, imSize, channels)

    tensorBatchData = flatBatch2Tensor(batchData, imSize, channels)
    if jitter:
        tensorBatchData = jitterBatch(tensorBatchData,cropSize,imSize)
    if randTransform:
        tensorBatchData = randTransformBatch(tensorBatchData)
    return tensorBatchData

def preProcessTestImages(batchData,imSize,cropSize,channels,rescale=True,stretch=False,
                     means=None,stds=None,stretchLow=None,stretchHigh=None):

    if rescale:
        batchData = rescaleBatch(batchData,means,stds,imSize,channels)
    if stretch:
        batchData = stretchBatch(batchData, stretchLow, stretchHigh, imSize, channels)
    tensorBatchData = flatBatch2Tensor(batchData,imSize,channels)

    tensorBatchData = extractCrops(tensorBatchData,cropSize,imSize)

    return tensorBatchData

def flatBatch2Tensor(batchData,imSize,channels):
    splitByChannel = [batchData[:,(chan*imSize**2):((chan+1)*imSize**2)].reshape((-1,imSize,imSize,1)) \
                      for chan in range(channels)]
    tensorBatchData = np.concatenate(splitByChannel,3)
    
    return tensorBatchData


def rescaleBatch(batchData,means,stds, imSize, channels):
    for chan in range(channels):
        batchData[:,(chan*imSize**2):((chan+1)*imSize**2)] = \
            (batchData[:,(chan*imSize**2):((chan+1)*imSize**2)] - means[chan]) / stds[chan]
    return batchData


def stretchBatch(batchData, lowerPercentile, upperPercentile, imSize, channels):
    for chan in range(channels):
        for i in range(len(batchData)):
            batchData[i, (chan * imSize ** 2):((chan + 1) * imSize ** 2)] = \
                stretchVector(batchData[i, (chan * imSize ** 2):((chan + 1) * imSize ** 2)],
                              lowerPercentile, upperPercentile)
    return batchData

def stretchVector(vec, lowerPercentile, upperPercentile):
    minVal = np.percentile(vec, lowerPercentile)
    maxVal = np.percentile(vec, upperPercentile)
    vec[vec > maxVal] = maxVal
    vec = vec - minVal
    if (maxVal-minVal)>1.:
        vec = vec / (maxVal - minVal)

    return vec

def jitterBatch(batchData,cropSize,imSize):
    batchSize,x,y,channels = batchData.shape
    croppedBatch = np.zeros((batchSize,cropSize,cropSize,channels),dtype=batchData.dtype)
    jitterPix = imSize-cropSize
    for i in range(batchSize):
        offset = np.random.randint(0,jitterPix,2)
        croppedBatch[i,:,:,:] = batchData[i,offset[0]:x-(jitterPix-offset[0]),
                                            offset[1]:y-(jitterPix-offset[1]),:]  
    return croppedBatch

def extractCrops(batchData,cropSize,imSize):
    batchSize,x,y,channels = batchData.shape
    crops = 5
    croppedBatch = np.zeros((batchSize,crops,cropSize,cropSize,channels),dtype=batchData.dtype)
    jitterPix = imSize-cropSize


    for i in range(batchSize):

     #center crop
        offset = [jitterPix/2,jitterPix/2]
        croppedBatch[i,0,:,:,:] = batchData[i,offset[0]:x-(jitterPix-offset[0]),
                                            offset[1]:y-(jitterPix-offset[1]),:]
    #left top crop
        offset = [0,0]
    #for i in range(batchSize):
        croppedBatch[i,1,:,:,:] = batchData[i,offset[0]:x-(jitterPix-offset[0]),
                                            offset[1]:y-(jitterPix-offset[1]),:]
    #left bottom crop
        offset = [0,jitterPix]
    #for i in range(batchSize):
        croppedBatch[i,2,:,:,:] = batchData[i,offset[0]:x-(jitterPix-offset[0]),
                                            offset[1]:y-(jitterPix-offset[1]),:]
    #right top crop
        offset = [jitterPix,0]
    #for i in range(batchSize):
        croppedBatch[i,3,:,:,:] = batchData[i,offset[0]:x-(jitterPix-offset[0]),
                                            offset[1]:y-(jitterPix-offset[1]),:]
    #right bottom crop
        offset = [jitterPix,jitterPix]
    #for i in range(batchSize):
        croppedBatch[i,4,:,:,:] = batchData[i,offset[0]:x-(jitterPix-offset[0]),
                                            offset[1]:y-(jitterPix-offset[1]),:]
    return croppedBatch


def randTransformBatch(croppedBatchData):
    for i in range(len(croppedBatchData)):
        if np.random.choice([True, False]):
            croppedBatchData[i,:,:,:] = np.flipud(croppedBatchData[i,:,:,:])
        if np.random.choice([True, False]):
            croppedBatchData[i,:,:,:] = np.fliplr(croppedBatchData[i,:,:,:])
        croppedBatchData[i,:,:,:] = np.rot90(croppedBatchData[i,:,:,:], k=np.random.randint(0,3))   
    return croppedBatchData