# CleaningImages

Cleaning irrelevant images from dataset.
Dataset needs some clean operations if you get your images with scraping. This repo contains scripts that deletes irrelevant images and creates xml files for object detection.

# VGG

Gets all images under specified directory. If result of that image is not in wanted labels deletes image.
Needs Keras and Tensorflow/Theano.
NOTE: It will download VGG weights(~500MB). 

# YOLO

Object detection algorithms need xml files that contains object coordinates. For this task yolo can be used. Scripts under Yolo-Darkflow directory can be used after retraining yolo with your object in your dataset.
For retraining see: https://github.com/thtrieu/darkflow

#Label_image

Anaylzes images in the specified paths.Creates annotation files if image got a result greater than treshold otherwise deletes the image.

NOTE : Set options values according to your model before running. Set model to your configure file. 
       Set load to number of last .profile file in ckpt folder.

#Verify

Gets images from specified folder and runs model on them.Shows result on an opencv window. Press any key to run next image.

NOTE : Set options values according to your model before running.

#Xml_bounds_test

Gets xml files and images from specified folders. Gets coordinates from xml files and shows on the image as opencv window.
Can bu used as checker for created xml files.
