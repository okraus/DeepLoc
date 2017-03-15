#! /bin/bash

echo "downloading all DeepLoc datasets"
wget http://spidey.ccbr.utoronto.ca/~okraus/DeepLoc_full_datasets.zip
unzip DeepLoc_full_datasets.zip
rm DeepLoc_full_datasets.zip

echo "downloading all DeepLoc pretrained model"
wget http://spidey.ccbr.utoronto.ca/~okraus/pretrained_DeepLoc.zip
unzip pretrained_DeepLoc.zip
rm pretrained_DeepLoc.zip
