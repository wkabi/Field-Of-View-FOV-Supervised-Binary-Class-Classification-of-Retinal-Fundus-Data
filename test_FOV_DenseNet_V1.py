## Testing FOV Model by employing Diagnos testing images
## Final result is saving the model on Gigantix.
## Written by: Waziha Kabir
## Started: July 1, 2021
## Last Modification Date: September 8, 2021

######################################### Import Libraries [START]
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy
import cv2
import gc
import copy
import tensorflow as tf
import csv
import keras

from keras.preprocessing import image
from keras.models import load_model
######################################### Import Libraries [END]

######################################### Calling images using .csv file [START]
## Location of .csv files
PATH_CSV = './CSV_FOV/CSV_FOV_Test_V1/'

## Lists of .csv files (if more than one dataset)
filelist =  os.listdir(PATH_CSV)
print('** Name of the csv files:', filelist)

## Read .csv files ('0' means first file)
dataset_diagnos = pd.read_csv(PATH_CSV + filelist[0], index_col = False)

## Show 1 .csv as array
#filename_diagnos = np.array(dataset_diagnos[['image_name','Dataset']][0:10])
filename_diagnos = np.array(dataset_diagnos[['x']][:])

## Call images and labels
test_x_img = filename_diagnos[:,0].astype('str')

## Show 1:1 
print('** Name of Images in array format:', test_x_img)
######################################### Calling images using .csv file [END]

######################################### Import Diagnos Training Data [START]
## Testing Data
import h5py
with h5py.File("./prepro_FOV_Test_V1/Xy_Origin_FOV_Test_V1.h5", "r") as f:
#     print(f.file)
    X_test = np.array(f['X'])
    #y_train = np.array(f["y"])

#print('** total number of training Label:', sum(y_train[:]))

print('** Shape of X_test: ', X_test.shape)
#print('** Shape of y_train: ', y_train.shape)
######################################## Import Diagnos Training Data [END]

test_X = np.array(X_test)
test_X = test_X / 255
######################################## Preprocessing images [END]

######################################## Load Model [START]
#Filename = "model/weights.165-0.099-0.9920.hdf5
#model_T = load_model('./model_FOV_DenseNet_v1/weights.01-0.0523-0.9871.hdf5') #Model 1
#model_T = load_model('./model_FOV_DenseNet_v1/weights.02-0.0326-0.9897.hdf5') #Model 2
#model_T = load_model('./model_FOV_DenseNet_v1/weights.03-0.0291-0.9921.hdf5') #Model 3
#model_T = load_model('./model_FOV_DenseNet_v1/weights.06-0.0276-0.9927.hdf5') #Model 4
#model_T = load_model('./model_FOV_DenseNet_v1/weights.10-0.0242-0.9921.hdf5') #Model 5
model_T = load_model('./model_FOV_DenseNet_v1/weights.18-0.0204-0.9940.hdf5') #Model 6
######################################## Load Model [END]

######################################## Prediction [START]
y_T = model_T.predict(test_X)

print('Model_2 predictions: ', y_T[:,0])
print('Model_2 predictions rounded: ', np.around(y_T[:,0]))

#a_inv = np.around(y_[:,0])
#print('inverted', np.invert(a_inv))
######################################## Prediction [END]

print('** Prediction is completed successfully!!')

######################################## Saving Prediction results in .csv file [START]
#with open('./Results/Results_FOV_DenseNet/'+'Result_Mdl1_FOV_DenseNet_V1.csv', mode='w') as result_file:
#with open('./Results/Results_FOV_DenseNet/'+'Result_Mdl2_FOV_DenseNet_V1.csv', mode='w') as result_file:
#with open('./Results/Results_FOV_DenseNet/'+'Result_Mdl3_FOV_DenseNet_V1.csv', mode='w') as result_file:
#with open('./Results/Results_FOV_DenseNet/'+'Result_Mdl4_FOV_DenseNet_V1.csv', mode='w') as result_file:
#with open('./Results/Results_FOV_DenseNet/'+'Result_Mdl5_FOV_DenseNet_V1.csv', mode='w') as result_file:
with open('./Results/Results_FOV_DenseNet/'+'Result_Mdl6_FOV_DenseNet_V1.csv', mode='w') as result_file:
    result_writer = csv.writer(result_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    
    result_writer.writerow(['image_id', ' predicted_prob',  ' predicted_label'])
    #result_writer.writerow(['image_id',  ' predicted_label'])
    result_writer.writerow([])
    for i in range(4553):
      result_writer.writerow([str(test_x_img[i]), y_T[i,0], np.around(y_T[i,0])])
      #result_writer.writerow([str(train_x_img[i]), np.around(y_T[i,0])])

print("** Predicted results are saved in './Results/Results_FOV_DenseNet/' directory")
######################################## Saving Prediction results in .csv file [END]


