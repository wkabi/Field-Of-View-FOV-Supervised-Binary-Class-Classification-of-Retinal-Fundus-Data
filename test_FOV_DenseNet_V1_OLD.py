## Final result is the prediction of FOV using Diagnos FOV data and model trained on Gigantix. 
## Re-written by: Waziha Kabir
## Started: July 30, 2021
## Last Modification Date: Aug 2, 2021

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
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD,Adam
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

######################################## Function: Check image shape [START]
def shape_check(img_shape_list, *args):
    """Check image shape by target shapes
    Args:
        img_shape_list: a list of image shape
        *args: target shapes
    Return:
        Boolean
    """
    
    count_list = []
    for arg in args:
        count = 0
#         print(arg)
        
        if arg not in img_shape_list:
            print('Target shape{} is not in shape list'.format(arg))
            
        for img_shape in img_shape_list:
            if img_shape == arg:
                count += 1
    
        count_list.append(count)
        
    print('Shapes checked {}'.format(sum(count_list)))   
    if sum(count_list) != len(img_shape_list):
        print('There are extra shapes in shape list')
        return False
    else: 
        return True
######################################## Function: Check image shape [END]

######################################## Function: Resize 'Single' image [START]
def image_resize(img, shape):
    """Resize different image sizes to shape by clip and cv2.resize()
    Args:
        img: source image array
        shape: target resize shape
    Return:
        type array, resized image
    """
    scale = img.shape[0] / img.shape[1]
    #print('Ratio of image width and height:', scale)
    if 0.9 <= scale < 1:
        return cv2.resize(img,shape,interpolation = cv2.INTER_NEAREST)
    else:
        length = img.shape[1]
        width = img.shape[0]
        
        # find attention area edge by moving check line i = img[:,i]
        index = 0
        index_move = True
        while index_move:
            index += 1
            if np.sum(img[:,index] == 0) != width*3:
                index_move = False
#         print(index,index_move)       
        # clip        
        img = img[:, index : length-index]
        
        return cv2.resize(img,shape,interpolation = cv2.INTER_NEAREST)
######################################## Function: Resize 'Single' image [END]

######################################## Function: Enhance 'Single' image using 4 methods [START]
def image_enhance(img,method):
    """Enhance image by 'origin', 'clahe', 'lac'
    Args:
        img: a image array with shape (299,299,3)
        method: 'origin','clahe','gray','LAC'
    Return:
        enhanced image.
    """
    def mask_image(img):
        SCALE = 299 // 2
        mask = np.zeros((299,299,3))
        cv2.circle(mask,(SCALE,SCALE),int(SCALE * 0.95),(1,1,1),-1,8,0)
        return (img * mask + 128 * (1-mask)).astype(np.uint8)
    
    if img.shape != (299,299,3):    
        print('Image shape is wrong, required (299,299,3)')
    else:             
        if method == 'origin':
            return mask_image(img)
        
        elif method == 'clahe':
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))  
            img_1 = clahe.apply(img[:,:,0]) * 1
            img_2 = clahe.apply(img[:,:,1]) * 1
            img_3 = clahe.apply(img[:,:,2]) * 1
            return mask_image(cv2.merge([img_1,img_2,img_3]))
        
        elif method == 'LAC':
            return mask_image(cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), 5), -4, 128))
        
        elif method in ['grey','gray']:        
            gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            return mask_image(cv2.merge([gray,gray,gray]))

        else:
            print('Method not supported!')
######################################## Function: Enhance 'Single' image using 4 methods [END]

######################################## Function: Enhance 'Multiple' image using 4 methods [START]
def image_load(image_path, images, method):
    """Load images for directory.
    Args:
        image_path: path of directory
        images: images name list. type: list
        method: enhanced mathod
    Return:
        list of image arrays.
    """
    X = []
    image_shape_list = []
    count = 0
    for name in images:
        print(name)
        #img = cv2.imread(image_path+name+'.jpg')
        img = cv2.imread(image_path+name)
        img = image_resize(img, (299,299))
        img = image_enhance(img, method = method)
        
        X.append(img)
        image_shape_list.append(img.shape)
        count += 1
        print("{:.2f}% images processed".format(count/len(images)*100), end='\r')
    
    print('\n')
    print('{} images have been loaded and preprocessed'.format(count))
    #return img
    if shape_check(image_shape_list, (299,299,3)):
        return X
    else:
#         print(shape_check(image_shape_list, (299,299,3)))
        print('Maybe some wrong shapes exist!')
######################################## Function: Enhance 'Multiple' image using 4 methods [END]

######################################## Preprocessing images [START]
## Location of test images of 'unprocessed_images' folder
test_X = image_load('/home/mracine/FOV_Database/Test_FOV_Dataset/', test_x_img, 'origin')

test_X = np.array(test_X)
test_X = test_X / 255
######################################## Preprocessing images [END]

######################################## Load Model [START]
#Filename = "model/weights.165-0.099-0.9920.hdf5
model_T = load_model('./model_FOV_DenseNet_v1/weights.01-0.0523-0.9871.hdf5') #Model 1
#model_T = load_model('./model_FOV_DenseNet_v1/weights.02-0.0326-0.9897.hdf5') #Model 2
#model_T = load_model('./model_FOV_DenseNet_v1/weights.03-0.0291-0.9921.hdf5') #Model 3
#model_T = load_model('./model_FOV_DenseNet_v1/weights.06-0.0276-0.9927.hdf5') #Model 4
#model_T = load_model('./model_FOV_DenseNet_v1/weights.10-0.0242-0.9921.hdf5') #Model 5
#model_T = load_model('./model_FOV_DenseNet_v1/weights.18-0.0204-0.9940.hdf5') #Model 6
######################################## Prediction [START]
#y_1 = model_1.predict(test_X)

#print('Model_1 predictions: ', y_1[:,0])
#print('Model_1 predictions rounded: ', np.around(y_1[:,0]))

y_T = model_T.predict(test_X)

print('Model_2 predictions: ', y_T[:,0])
print('Model_2 predictions rounded: ', np.around(y_T[:,0]))

#a_inv = np.around(y_[:,0])
#print('inverted', np.invert(a_inv))
######################################## Prediction [END]

print('** Prediction is completed successfully!!')

######################################## Saving Prediction results in .csv file [START]
with open('./Results/Results_FOV_DenseNet/'+'Result_Mdl1_FOV_DenseNet_V1.csv', mode='w') as result_file:
#with open('./Results/Results_FOV_DenseNet/'+'Result_Mdl2_FOV_DenseNet_V1.csv', mode='w') as result_file:
#with open('./Results/Results_FOV_DenseNet/'+'Result_Mdl3_FOV_DenseNet_V1.csv', mode='w') as result_file:
#with open('./Results/Results_FOV_DenseNet/'+'Result_Mdl4_FOV_DenseNet_V1.csv', mode='w') as result_file:
#with open('./Results/Results_FOV_DenseNet/'+'Result_Mdl5_FOV_DenseNet_V1.csv', mode='w') as result_file:
#with open('./Results/Results_FOV_DenseNet/'+'Result_Mdl6_FOV_DenseNet_V1.csv', mode='w') as result_file:
    result_writer = csv.writer(result_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    
    result_writer.writerow(['image_id', ' predicted_prob',  ' predicted_label'])
    #result_writer.writerow(['image_id',  ' predicted_label'])
    result_writer.writerow([])
    for i in range(4553):
      result_writer.writerow([str(test_x_img[i]), y_T[i,0], np.around(y_T[i,0])])
      #result_writer.writerow([str(train_x_img[i]), np.around(y_T[i,0])])

print("** Predicted results are saved in './Results/Results_FOV_DenseNet/' directory")
######################################## Saving Prediction results in .csv file [END]


