# DeepLoc
This repository contains the code, pretrained models, and datasets for the paper:
"Automated analysis of high-content microscopy data with deep learning"
Kraus, O.Z., Grys, B.T., Ba, J., Chong, Y., Frey, B.J., Boone, C., & Andrews, B.J.
Molecular Systems Biology

DeepLoc training scripts
=========================

This source tree was distributed as part of the supplementary software 
release for DeepLoc. Users of DeepLoc are encouraged to instead use the 
latest source code and binaries at

   https://github.com/okraus/DeepLoc

The commands below assume starting from the 'DeepLoc_Supplemental_Software' subdirectory.



REQUIREMENTS
------------

for training and evaluation scripts:

Python 2.7+ 64-bit
   http://www.python.org/getit/

CUDA 8.0+ SDK (for GPU support)
   https://developer.nvidia.com/cuda-downloads
cuDNN 5.1 (for GPU support)
   https://developer.nvidia.com/cudnn

Tensorflow v1.0+
    https://www.tensorflow.org/install/install_linux

You can use the following command to install the dependencies (other than tensorflow):

  pip install -r requirements.txt

or if running the Anaconda Python distribution use to start an environment with the dependencies:

  conda env create -f environment.yml


Tensorflow 0.10+
    https://www.tensorflow.org/versions/r0.10/get_started/os_setup#virtualenv_installation
    - developed on tensorflow version 0.10
    - briefly tested on version 0.12


GETTING THE FULL DATA SETS
--------------------------
Due to the 150MB size limit on Supplementary Software, this package contains only
small subset (datasets_small) of the training and testing data, and not all scripts will run
without the missing data.

To download and unzip the datasets and pretrained model, please run:

  bash download_datasets.sh

Otherwise, the datasets are available at:

  http://spidey.ccbr.utoronto.ca/~okraus/DeepLoc_full_datasets.zip

and the pretrained model is available at:

  http://spidey.ccbr.utoronto.ca/~okraus/pretrained_DeepLoc.zip

to download the full data sets and include them in the 'datasets' subdirectory.


TRAINING DeepLoc on Chong et al., 2015 DATA (CELL, doi:10.1016/j.cell.2015.04.051)
----------------------------------------------------------------------------------

To train DeepLoc on the Chong et al. dataset run:

   python DeepLoc_train.py -logdir path/to/log-directory

   - the argument passed to -logdir indicates where to save the resulting models
     and model predictions (a good default is "./logs")
   - By default DeepLoc will train on the limited datasets in the ./datasets directory
     unless the full datasets are downloaded as described above and stored in ./datasets
   - By default, models are saved every 500 iterations, and a test batch is evaluated every 50 iterations

To evaluate the performance of different DeepLoc checkpoints run:

    python DeepLoc_eval.py -logdir path/to/log-directory

    - the argument to -logdir should be the same path used for training
    - adds a python cPickle file called "test_acc_deploy_results.pkl" in the path/to/log-directory
      including training and test performance (accuracy and test values) for the full datasets

VISUALIZING DeepLoc GRAPH AND TRAINING PERFORMANCE
--------------------------------------------------

The DeepLoc model and training performance can be visualized using Tensorboard
(https://www.tensorflow.org/how_tos/summaries_and_tensorboard/)

To initialize a tensorboard session, run:

    tensorboard --logdir=path/to/log-directory

once TensorBoard is running, navigate your web browser to localhost:6006 to view the TensorBoard


DEPLOYING DeepLoc TO SAMPLE IMAGE FROM ENTIRE SCREEN
----------------------------------------------------

DeepLoc can be deployed to an entire automated microscopy screen using the demo in:

    python DeepLoc_eval_sample_image.py
    - assumes model saved in './pretrained_models/model.ckpt-9500'
    - output stored as csv file in './sample_image'

VISUALIZING DeepLoc CLASSES AND FEATURES
----------------------------------------

Patterns that maximally activate DeepLoc output classes can be visualized using:

    python DeepLoc_visualize_classes.py -loc_ckpt path/to/trained_model
    - use './pretrained_models/model.ckpt-9500' as path/to/trained_model
    - output stored in './output_figures/generated_cells.png'


TRANSFERING DeepLoc TO wt2017 and SWAT_RFP DATASETS
---------------------------------------------------

DeepLoc can be loaded and fine-tuned on the wt2017 dataset using:

    python DeepLoc_transfer_wt2017.py
    - assumes model saved in './pretrained_models/model.ckpt-9500'
    -output stored in './logs/transfer_wt2017

and fine-tuned on the SWAT_RFP dataset using:

    python DeepLoc_transfer_SWAT_RFP.py
    - assumes model saved in './pretrained_models/model.ckpt-9500'
    -output stored in './logs/transfer_SWAT_RFP
    