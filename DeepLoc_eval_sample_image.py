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
import os
import Load_GR
from PIL import Image
import preprocess_images as procIm
import numpy as np
import pandas as pd
import copy



import argparse
parser = argparse.ArgumentParser(description='Evaluate sample image using DeepLoc model')
parser.add_argument("-l","--logdir",action="store",dest="logdir",help="directory to save models",
                    default='./pretrained_DeepLoc/pretrained_models/model.ckpt-5000')
parser.add_argument("-o", "--output-folder", action="store", dest="outputdir", help="directory to store results",
                    default='./sample_image')
args = parser.parse_args()
print 'log dir:',args.logdir,'out dir:',args.outputdir


locNetCkpt = args.logdir
output_dir = args.outputdir

if not os.path.exists(locNetCkpt+'.meta'):
    raise NameError('please download pretrained model')


class screenClass:
    """
    This class manages the the background threads needed to fill
        a queue full of data.
    """

    def __init__(self, screen='sample_image'):

        self.sql_col_names = ['OrigIndex','cscreen', 'ImageNumber', 'Image_FileName_GFP', 'Image_FileName_RFP',
                              'Image_FileName_overlay', 'nwellidf',
                              'cells_Location_Center_X', 'cells_Location_Center_Y', 'cells_AreaShape_Area',
                              'cells_AreaShape_Eccentricity',
                              'cells_AreaShape_EulerNumber', 'cells_AreaShape_Extent', 'cells_AreaShape_FormFactor',
                              'cells_AreaShape_MajorAxisLength', 'cells_AreaShape_MinorAxisLenght',
                              'cells_AreaShape_Orientation',
                              'cells_AreaShape_Perimeter', 'cells_AreaShape_Solidarity',
                              'cells_Intensity_IntegratedIntensity_GFP', 'cells_Intensity_MeanIntensity_GFP',
                              'cells_Intensity_StdIntensity_GFP', 'cells_Intensity_MinIntensity_GFP',
                              'cells_Intensity_MaxIntensity_GFP', 'cells_Intensity_IntegratedIntensityE_GFP']

        self.localizationTerms = ['ACTIN', 'BUDNECK', 'BUDTIP', 'CELLPERIPHERY', 'CYTOPLASM',
                                  'ENDOSOME', 'ER', 'GOLGI', 'MITOCHONDRIA', 'NUCLEARPERIPHERY',
                                  'NUCLEI', 'NUCLEOLUS', 'PEROXISOME', 'SPINDLE', 'SPINDLEPOLE',
                                  'VACUOLARMEMBRANE', 'VACUOLE', 'DEAD', 'GHOST']


        self.basePath = './'+screen+'/'
        self.sql_data = pd.read_csv(self.basePath+'SQL_data.csv')
        self.sql_data.columns = self.sql_col_names

        GFP_images = np.unique(self.sql_data['Image_FileName_GFP'])
        GFP_images.sort()
        self.wells = np.unique([seq[:-2] for seq in GFP_images])

        self.cropSize = 60

        self.imSize = 64
        self.numClasses = 19
        self.numChan = 2


    def processWell(self, well):

        ### load from jpeg instead because HOwt flex files were stored in 8bit ###
        ###### switch back to flex, rescale to 0-1 by stretching

        curFlex = Image.open(self.basePath + well + '.flex')

        G, R = Load_GR.load(curFlex)
        G_arrays = Load_GR.convert(G)
        R_arrays = Load_GR.convert(R)

        MAX_CELLS = 1200
        croppedCells = np.zeros((MAX_CELLS, self.imSize ** 2 * 2))
        coordUsed = np.zeros((MAX_CELLS, 2))
        intensityUsed = np.zeros((MAX_CELLS, 5))
        ind = 0
        wellNames = []
        for frame in range(1, 8, 2):

            G_array = G_arrays[frame/2]
            R_array = R_arrays[frame/2]

            curCoordinates = self.sql_data[self.sql_data['Image_FileName_GFP'] == well + '_' + str(frame)][
                ['cells_Location_Center_X',
                 'cells_Location_Center_Y']]

            curIntensity = self.sql_data[self.sql_data['Image_FileName_GFP'] == well + '_' + str(frame)][
                ['cells_Intensity_IntegratedIntensity_GFP',
                 'cells_Intensity_MeanIntensity_GFP', 'cells_Intensity_StdIntensity_GFP',
                 'cells_Intensity_MinIntensity_GFP', 'cells_Intensity_MaxIntensity_GFP']]
            coord = 0

            while coord < len(curCoordinates):
                cur_y, cur_x = curCoordinates.values[coord]
                # delete frame/2 because image is now single frame
                if cur_x - self.imSize / 2 > 0 and cur_x + self.imSize / 2 < G_array.shape[
                    0] and cur_y - self.imSize / 2 > 0 and cur_y + self.imSize / 2 < G_array.shape[1]:

                    croppedCells[ind, : self.imSize ** 2] = (
                    G_array[int(np.floor(cur_x - self.imSize / 2)):int(np.floor(cur_x + self.imSize / 2)),
                    int(np.floor(cur_y - self.imSize / 2)):int(np.floor(cur_y + self.imSize / 2))]).ravel()

                    croppedCells[ind, self.imSize ** 2 :] = (
                    R_array[int(np.floor(cur_x - self.imSize / 2)):int(np.floor(cur_x + self.imSize / 2)),
                    int(np.floor(cur_y - self.imSize / 2)):int(np.floor(cur_y + self.imSize / 2))]).ravel()

                    coordUsed[ind, :] = [cur_y, cur_x]
                    intensityUsed[ind, :] = curIntensity.values[coord, :]
                    coord += 1
                    ind += 1
                    wellNames.append(well + '_' + str(frame / 2))

                else:
                    coord += 1
                if ind > (MAX_CELLS-1):
                    break
            if ind > (MAX_CELLS-1):
                break

        curCroppedCells = croppedCells[:ind]
        intensityUsed = intensityUsed[:ind]
        coordUsed = coordUsed[:ind]

        ### stretch flex files to be between 0  - 1
        stretchLow = 0.1  # stretch channels lower percentile
        stretchHigh = 99.9  # stretch channels upper percentile
        processedBatch = procIm.preProcessTestImages(curCroppedCells,
                                                     self.imSize, self.cropSize, self.numChan,
                                                     rescale=False, stretch=True,
                                                     means=None, stds=None,
                                                     stretchLow=stretchLow, stretchHigh=stretchHigh)

        # print(well+'_'+str(frame))
        return processedBatch, coordUsed, intensityUsed, wellNames




def proccessCropsLoc(processedBatch,predicted_y,inputs,is_training,sess):
    crop_list = np.zeros((len(processedBatch), 5, 19))
    for crop in range(5):
        images = processedBatch[:, crop, :, :, :]
        tmp = copy.copy(sess.run([predicted_y], feed_dict={inputs: images, is_training: False}))
        # print(tmp)
        crop_list[:, crop, :] = tmp[0]

    mean_crops = np.mean(crop_list, 1)
    return mean_crops

def eval():

    #####################
    ### LOAD NETWORKS ###
    #####################

    #LOCALIZATION
    loc = tf.Graph()
    with loc.as_default():
        loc_saver = tf.train.import_meta_graph(locNetCkpt+'.meta')
    locSession = tf.Session(graph=loc)
    loc_saver.restore(locSession, locNetCkpt)

    pred_loc = loc.get_tensor_by_name(u'softmax:0')
    input_loc = loc.get_tensor_by_name(u'input:0')
    is_training_loc = loc.get_tensor_by_name(u'is_training:0')

    ###################################################################################################################


    localizationTerms = ['ACTIN', 'BUDNECK', 'BUDTIP', 'CELLPERIPHERY', 'CYTOPLASM',
                     'ENDOSOME', 'ER', 'GOLGI', 'MITOCHONDRIA', 'NUCLEARPERIPHERY',
                     'NUCLEI', 'NUCLEOLUS', 'PEROXISOME', 'SPINDLE', 'SPINDLEPOLE',
                     'VACUOLARMEMBRANE', 'VACUOLE', 'DEAD', 'GHOST']

    col_names_output = ['x_loc', 'y_loc', 'cells_Intensity_IntegratedIntensity_GFP',
                    'cells_Intensity_MeanIntensity_GFP', 'cells_Intensity_StdIntensity_GFP',
                    'cells_Intensity_MinIntensity_GFP', 'cells_Intensity_MaxIntensity_GFP'] + localizationTerms

    allPred = None

    curScreenClass = screenClass(screen='sample_image')
    processedBatch, coordUsed, intensityUsed, wellNames = curScreenClass.processWell('plate01/007020000')


    del allPred
    allPred = pd.DataFrame(np.zeros((curScreenClass.sql_data.shape[0],
                                     len(col_names_output))), columns=col_names_output)
    allPred_ind = 0

    wellNamesAll = []

    wellNamesAll.append(wellNames)

    predictedBatch_Loc = proccessCropsLoc(processedBatch=processedBatch, predicted_y=pred_loc,
                                   inputs=input_loc,is_training=is_training_loc, sess=locSession)

    allPred.iloc[allPred_ind:allPred_ind + len(predictedBatch_Loc), :] = np.hstack((
        coordUsed, intensityUsed, predictedBatch_Loc))
    allPred_ind += len(predictedBatch_Loc)


    allPred = allPred.iloc[:allPred_ind, :]
    allPred['well'] = np.hstack(wellNamesAll)


    locCkptBasename = os.path.basename(locNetCkpt)

    allPred.to_csv(output_dir+'/'+locCkptBasename+'_localization_pred_v1.csv')

    locSession.close()


def main(_):
    eval()


if __name__ == '__main__':
    tf.app.run()
