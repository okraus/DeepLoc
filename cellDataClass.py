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



import h5py
import numpy as np


class Data:
    def __init__(self,
                 folder,
                 keys2fetch,
                 batchSize,
                 ):
        self.numData = 0
        self.batchSize = batchSize
        self.folder = folder
        self.h5data = h5py.File(self.folder, 'r')
        self.keys2fetch = keys2fetch
        h5keys = self.h5data.keys()
        self.groupedData = {}
        for key in keys2fetch: self.groupedData[key] = []
        for key in h5keys:
            if any(x in key for x in keys2fetch):
                curInd = [x in key for x in keys2fetch]
                if curInd[0]:
                    self.numData += len(self.h5data[key])
                curKey = keys2fetch[curInd.index(True)]
                self.groupedData[curKey].append(int(key[len(curKey):]))
        for key in keys2fetch: self.groupedData[key].sort()

        self.startInd = 0
        self.stopInd = self.numData
        self.curInd = self.startInd

        assert batchSize<self.numData, "batchSize larger than dataset; batchSize: "+str(batchSize)+" dataSize: "+str(self.numData)

        self.h5chunkSize = len(self.h5data[keys2fetch[0] + '1'])
        self.keySizes = {}
        for key in keys2fetch: self.keySizes[key] = self.h5data[key + '1'].shape[1]

        self.returnArrays = {}
        for key in keys2fetch:
            self.returnArrays[key] = np.zeros((self.batchSize, self.keySizes[key]), dtype=np.float32)

    def getBatch(self):

        if (self.curInd + self.batchSize) >= self.stopInd:
            self.curInd = self.startInd

        startDsetNum = self.curInd / self.h5chunkSize + 1
        startDsetInd = self.curInd % self.h5chunkSize
        endDsetNum = (self.curInd + self.batchSize) / self.h5chunkSize + 1

        for key in self.keys2fetch:
            curInd = 0
            curDset = startDsetNum
            curDsetInd = startDsetInd
            while curInd < self.batchSize:
                dsetShape = self.h5data[key + str(curDset)].shape
                self.returnArrays[key][curInd:min(dsetShape[0] - curDsetInd, self.batchSize + curDsetInd), :] = \
                    self.h5data[key + str(curDset)][curDsetInd:min(dsetShape[0], self.batchSize + curDsetInd), :]
                curDset += 1
                curDsetInd = 0
                curInd += min(dsetShape[0] - curDsetInd, self.batchSize)

        self.curInd += self.batchSize

        return self.returnArrays
