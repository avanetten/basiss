#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 17:26:51 2018

@author: avanetten
"""

from __future__ import print_function

import numpy as np
import argparse
import time
import os
import datetime
import gdal
import pandas as pd
import cv2
#import json
#import tensorflow as tf

from keras.models import Model
from keras.layers import (Dense, Dropout, Activation, Flatten, Reshape, 
                          Lambda, Convolution2D, Conv2D, MaxPooling2D, 
                          UpSampling2D, Input, merge, Concatenate, 
                          concatenate, Conv2DTranspose)
from keras.callbacks import ModelCheckpoint, EarlyStopping
                        #,LearningRateScheduler
from keras.optimizers import SGD, Adam, Adagrad
from keras import backend as K
import keras.utils.vis_utils

#from keras.regularizers import l2
#from keras.utils.np_utils import to_categorical
#from keras.initializers import Identity
#from keras.layers.core import Permute
#from keras.layers.normalization import BatchNormalization


###############################################################################
### Jaccard
###############################################################################
def jaccard_coef(y_true, y_pred, smooth=1e-12):
    '''https://www.kaggle.com/drn01z3/end-to-end-baseline-with-u-net-keras
    # __author__ = Vladimir Iglovikov'''
    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return K.mean(jac)

###############################################################################
def jaccard_coef_int(y_true, y_pred, smooth=1e-12):
    '''https://www.kaggle.com/drn01z3/end-to-end-baseline-with-u-net-keras
    # __author__ = Vladimir Iglovikov'''
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))

    intersection = K.sum(y_true * y_pred_pos, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return K.mean(jac)

###############################################################################
###############################################################################
### Manually define metrics
# define metrics from https://github.com/fchollet/keras/blob/master/keras/metrics.py
#   since for some reason keras can't always import them
###############################################################################    
def dice_coeff(y_true, y_pred):
    y_true_flat = K.flatten(y_true)
    y_pred_flat = K.flatten(y_pred)
    intersect = K.sum(y_true_flat * y_pred_flat)
    return (2. * intersect) / (K.sum(y_true_flat) + K.sum(y_pred_flat))

###############################################################################
def dice_loss(y_true, y_pred):
    return -1. * dice_coeff(y_true, y_pred)
    
###############################################################################
def mse(y_true, y_pred):
    return K.mean(K.square(K.flatten(y_pred) - K.flatten(y_true)), axis=-1)

###############################################################################
def f1_score(y_true, y_pred):
    '''https://stackoverflow.com/questions/45411902/how-to-use-f1-score-with-keras-model'''

    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0

    # How many selected items are relevant?
    precision = c1 / c2

    # How many relevant items are selected?
    recall = c1 / c3

    # Calculate f1_score
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score 

###############################################################################
def f1_loss(y_true, y_pred):
    return 1. - f1_score(y_true, y_pred)


###############################################################################
### Data load functions
###############################################################################
def load_multiband_im(image_loc, nbands=3):
    '''use gdal to laod multiband files, else use cv2'''
    
    if nbands == 1:
        img = cv2.imread(image_loc, 0)
    elif nbands == 3:
        img = cv2.imread(image_loc, 1)
    else:
        #ingest 8 band image
        im8_raw = gdal.Open(image_loc)
        bandlist = []
        for band in range(1, im8_raw.RasterCount+1):
            srcband = im8_raw.GetRasterBand(band)
            band_arr_tmp = srcband.ReadAsArray()
            bandlist.append(band_arr_tmp)
        img = np.stack(bandlist, axis=2)
    
    return img


###############################################################################    
def load_files_universal(file_list_loc, nbands=3, n_classes=2,
                         output_size=(), max_len=10000000,
                         export_im_vis_arr=False, 
                         reshape_mask=False, super_verbose=False):
    '''Load image and mask files from file list
    any number of bands is acceptable
    file_list_loc has rows:
        [im_test_root, im_test_file, im_vis_file, mask_file, mask_vis_file])
    '''

    print ("\nLoading files from ", file_list_loc, "...")
    print ("  output_size:", output_size)

    df = pd.read_csv(file_list_loc, na_values='')
    
    #########################
    N = len(df)
    if len(output_size) > 0:
        h, w = output_size[0], output_size[1]
    else:
        # read in first row to get shape
        [im_test_root, im_test_file, im_vis_file, mask_file, \
                                             mask_vis_file] = df.iloc[0]
        sat = load_multiband_im(im_test_file, nbands=nbands)
        h, w, _ = sat.shape
    # set shapes
    im_shape = (N, h, w, nbands)
    mask_shape = (N, h, w, n_classes)
    #########################    

    # turning a list to an array often saturates memory, so create empty
    #   np arrays first
    im_arr = np.empty(im_shape, dtype=np.uint8)
    im_vis_arr = np.empty(im_shape, dtype=np.uint8)
    mask_arr = np.empty(mask_shape, dtype=np.uint8)
    name_arr = []
    for i,row in enumerate(df.values):
        
        if i > max_len:
            break
        
        [im_test_root, im_test_file, im_vis_file, mask_file, \
                                                  mask_vis_file] = row
        # load image
        sat = load_multiband_im(im_test_file, nbands=nbands)
        
        if (i % 50) == 0:
            print ("Loading file", i, "of", len(df))
            print ("   File path:", im_test_file)
            print ("   im.shape:", sat.shape)
            print ("   im.dtype:", sat.dtype)
            
        # usually, we'll skip im_vis_file
        if export_im_vis_arr:
            sat_vis = load_multiband_im(im_vis_file, nbands=nbands)
        else:
            sat_vis = []

        if n_classes <= 2:
            # if file name is null, set to ''
            #if np.isnan(mask_file):
            #    mask_file = ''
            # create mask
            if len(mask_file) > 0:
                # assume mask is has 1 layer!
                mask = load_multiband_im(mask_file, 1)  
            else:
                mask = np.zeros((sat.shape[0], sat.shape[1]))
        else:
            print ("Need write some more code for n_classes > 2!")
            break

        ################################
        # resize files, if desired
        if super_verbose:
            print ("mask.shape:", mask.shape)

        # set image size
        if len(output_size) > 0 :
            resize_size = output_size
        else:
            h, w = mask.shape[:2]
            resize_size = (h,w)
            
        # resize
        im_resize = cv2.resize(sat, resize_size)
        
        if export_im_vis_arr:
            im_vis_resize = cv2.resize(sat_vis, resize_size)
        else:
            im_vis_resize = []

        # make mask of appropriate dept (include background channel)
        if n_classes == 2:
            roads_resize = cv2.resize(mask, resize_size)
            bg_resize = np.array(np.ones(resize_size) \
                                 - roads_resize).astype(int)
            mask_resize = np.stack((bg_resize, roads_resize), axis=2)
            #mask_vis_resize = np.stack((bg_resize, roads_resize), axis=2)
        elif n_classes == 1:
            mask_resize = cv2.resize(mask, resize_size)
        else:
            print ("Need write some more code for n_classes > 2!")
            break
        #print "mask_resize.shape:", mask_resize
        ################################

        ################################
        # add to arrays
        name_arr.append(im_test_root)
        mask_arr[i] = mask_resize
        im_arr[i] = im_resize
        if export_im_vis_arr:
            im_vis_arr[i] = im_vis_resize
        # delete to save memory
        del row, sat, sat_vis, mask, im_resize, mask_resize

    # reshape mask, if desired
    if reshape_mask or n_classes==1:
        mask_arr0 = np.array(mask_arr)
        #print "mask_arr0.shape:", mask_arr0.shape
        N,h,w = mask_arr0.shape
        mask_arr = np.reshape(mask_arr0, (N,h,w,1))
    
    # reshape im_vis if only 1 band, since keras needs a tensor (N,h,w,nbands)
    if nbands == 1:
        tmp_arr = np.array(im_arr)
        N,h,w = tmp_arr.shape
        if super_verbose:
            print ("im_arr.shape:", tmp_arr.shape)
        im_arr = np.reshape(np.array(im_arr),  (N,h,w,1))
        im_vis_arr = np.reshape(np.array(im_vis_arr),  (N,h,w,1))
    
    
    print ("create np name arr..")
    name_arr = np.asarray(name_arr)
    print ("name_arr.shape:", name_arr.shape)

    return name_arr, im_arr, im_vis_arr, mask_arr


###############################################################################
def slice_ims(im_arr, mask_arr, names_arr, slice_x, slice_y, 
                    stride_x, stride_y,
                    pos_columns = ['idx', 'name', 'xmin', 'ymin', 'slice_x', 
                                   'slice_y', 'im_x', 'im_y'],
                    verbose=True, super_verbose=False):
    '''Slice images into patches, assume ground truth masks are present'''
    
    if verbose:
        print ("Slicing images and masks...")
    t0 = time.time()    
    mask_buffer = 0
    count = 0
    im_list, mask_list, name_list, pos_list = [], [], [], []
    nims,h,w,nbands = im_arr.shape
    for i, (im, mask, name) in enumerate(zip(im_arr, mask_arr, names_arr)):

        seen_coords = set()
        
        if verbose and (i % 100) == 0:
            print (i, "im_shape:", im.shape, "mask_shape:", mask.shape)
                
        # dice it up
        # after resize, iterate through image and bin it up appropriately
        for x in range(0, w - 1, stride_x):  
            for y in range(0, h - 1, stride_y): 
                
                xmin = min(x, w-slice_x)
                ymin = min(y, h - slice_y) 
                coords = (xmin, ymin)
                
                # check if we've already seen these coords
                if coords in seen_coords:
                    if super_verbose:
                        print ("coords already encountered ", \
                               "(xmin, ymin):", coords)
                    continue
                else:
                    seen_coords.add(coords)
                
                # check if we screwed up binning
                if (xmin + slice_x > w) or (ymin + slice_y > h):
                    print ("Improperly binned image, see slice_ims()")
                    return

                # get satellite image cutout
                im_cutout = im[ymin:ymin + slice_y, xmin:xmin + slice_x]
                
                ##############
                # skip if the whole thing is black
                if np.max(im_cutout) < 1.:
                    if super_verbose:
                        print ("Cutout null, skipping...")
                    continue
                else:
                    count += 1
                ###############
                
                # get mask cutout
                x1, y1 = xmin + mask_buffer, ymin + mask_buffer
                mask_cutout  = mask[y1:y1 + slice_y, x1:x1 + slice_x]
                
                if super_verbose:
                    print ("x, y:", x, y )
                    print ("  x_min, y_min:", xmin, ymin)
                    print ("  x_bounds:", xmin, xmin+slice_x, 
                                   "y_bounds:", ymin, ymin+slice_y)
                    print ("  im_cutout.shape:", im_cutout.shape)
                    print ("  mask_cutout.shape:", mask_cutout.shape)

                # set slice name
                name_full = str(i) + '_' + name + '_' \
                    + str(xmin) + '_' + str(ymin) + '_' + str(slice_x) \
                    + '_' + str(slice_y)  + '_' + str(w) + '_' + str(h)
                pos = [i, name, xmin, ymin, slice_x, slice_y, w, h]
                # add to arrays
                #idx_list.append(idx_full)
                name_list.append(name_full)
                im_list.append(im_cutout)
                mask_list.append(mask_cutout) 
                pos_list.append(pos)
    
    # convert to np arrays
    del im_arr
    del mask_arr
    #idx_out_arr = np.array(idx_list)
    name_out_arr = np.array(name_list)
    im_out_arr = np.array(im_list)
    mask_out_arr = np.array(mask_list)
    
    # create position datataframe
    df_pos = pd.DataFrame(pos_list, columns=pos_columns)
    df_pos.index = np.arange(len(df_pos))
    
    if verbose:
        print ("  im_out_arr.shape;", im_out_arr.shape)
        print ("  mask_out_arr.shape:", mask_out_arr.shape)
        print ("  mask_out_arr[0] == mask_out_arr[1]?:", 
                   np.array_equal(mask_out_arr[0], mask_out_arr[1]))
        print ("  time to slice arrays:", time.time() - t0, "seconds")
        
    return df_pos, name_out_arr, im_out_arr, mask_out_arr


###############################################################################
### Define model(s)
###############################################################################
def unet(input_shape, n_classes=2, kernel=3, loss='binary_crossentropy', 
         optimizer='SGD', data_format='channels_last'):
    '''
    https://arxiv.org/abs/1505.04597
    https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/master/train.py
    '''
    
    print ("UNET input shape:", input_shape) 
    #inputs = Input((input_size, input_size, n_channels))
    inputs = Input(input_shape)
    
    conv1 = Conv2D(32, (kernel, kernel), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (kernel, kernel), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (kernel, kernel), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (kernel, kernel), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (kernel, kernel), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (kernel, kernel), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (kernel, kernel), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (kernel, kernel), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (kernel, kernel), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (kernel, kernel), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    #up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
    conv6 = Conv2D(256, (kernel, kernel), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (kernel, kernel), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    #up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
    conv7 = Conv2D(128, (kernel, kernel), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (kernel, kernel), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    #up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    conv8 = Conv2D(64, (kernel, kernel), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (kernel, kernel), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    #up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
    conv9 = Conv2D(32, (kernel, kernel), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (kernel, kernel), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(n_classes, (1, 1), activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)
    
    if optimizer.upper() == 'SGD':
        opt_f = SGD()
    elif optimizer.upper == 'ADAM':
        opt_f = Adam()
    elif optimizer.upper == 'ADAGRAD':
        opt_f = Adagrad()
    else:
        print ("Unknown optimzer:", optimizer)
        return
    
    model.compile(optimizer=opt_f, loss=loss, 
                  metrics=[jaccard_coef, jaccard_coef_int, dice_coeff, 
                           'accuracy', 'mean_squared_error', f1_score])#,
                           #'precision', 'recall', 'f1score', 'mse'])

    print ("UNET Total number of params:", model.count_params() )     
    return model


###############################################################################
### Post-process
###############################################################################
def get_mask_tiles_from_name(df_pos, mask_predict_arr, name, 
                               verbose=False):
    
    '''We slice all input tiles and output a list of masks, images, and names
        df_pos_columns = ['idx', 'name', 'xmin', 'ymin', 'slice_x', 
                          'slice_y', 'im_x', 'im_y'],
    From this list of names, pull out all indexes that belong to a unique 
    image                   
    '''
    
    df_tmp = df_pos[df_pos['name'] == name]    
    idxs = df_tmp.index.values  #df_tmp['idx'].values  
    mask_tmp = mask_predict_arr[idxs]
    
    if verbose:
        print ("Gathering all tiles for image:", name)
        print ("  df_tmp:", df_tmp)
        print ("  idxs:", idxs)
        print ("  mask_tmp.shape:", mask_tmp.shape)
        print ("  mask_tmp[0] == mask_tmp[-1]?", np.array_equal(mask_tmp[0], \
               mask_tmp[-1]))

    return df_tmp, mask_tmp


###############################################################################
def post_process_mask(df_pos_, mask_arr_, n_classes=2, super_verbose=False):
    '''
    For a dataframe of image positions (df_pos_), and the tiles of that image
    (im_arr_), reconstruct the image mask
    '''
    
    # get image width and height
    w, h = df_pos_['im_x'].values[0], df_pos_['im_y'].values[0]
    
    # create numpy zeros of appropriate shape
    mask_raw = np.zeros((h,w))

    #  = create another zero array to record which pixels are overlaid
    overlay_count = np.zeros((h,w))

    # iterate through slices
    for i, (idx, item) in enumerate(df_pos_.iterrows()):
        #print (i, idx, item
        [idx, name, xmin, ymin, slice_x, slice_y, im_x, im_y] = item
        mask_slice = mask_arr_[i]
        if super_verbose:
            print ("item:", item)
            
        # for now, assume 2 channels, remove 0th (background) channel
        if n_classes != 2:
            print ("Need new code to handle n_channels != 2")
            return
        else:
            mask_slice_refine = mask_slice[:,:,1]
            
        x0, x1 = xmin, xmin + slice_x
        y0, y1 = ymin, ymin + slice_y

        # add mask to mask_raw
        mask_raw[y0:y1, x0:x1] += mask_slice_refine
        # update count
        overlay_count[y0:y1, x0:x1] += np.ones((slice_y, slice_x))

    # compute normalized mask
    # if overlay_count == 0, reset to 1
    overlay_count[np.where(overlay_count == 0)] = 1
                  
    mask_norm = np.divide(mask_raw, overlay_count)

    return mask_norm, mask_raw, overlay_count      
        

###############################################################################
def post_process_tiles(df_pos, mask_predict_arr, image_name, 
                       n_classes=2, super_verbose=False):
            
    '''We slice all test files and run through the classifier.  This function
    calls get_mask_times_from_name() and post_process_mask() for the given
    image name and creates total masks
    '''
            
    df_tmp, mask_tmp = get_mask_tiles_from_name(df_pos, mask_predict_arr, 
                                                image_name, 
                                                verbose=super_verbose)
    mask_norm, mask_raw, overlay_count = post_process_mask(df_tmp, mask_tmp, 
                                  n_classes=n_classes, 
                                  super_verbose=super_verbose)
    
    return mask_norm, mask_raw, overlay_count


###############################################################################
def post_process_tiles_all(df_pos, mask_predict_arr, out_dir, out_dir_vis='',
                       out_dir_raw='', out_dir_count='', 
                       n_classes=2, mask_max=255., verbose=False,
                       super_verbose=False):
            
    '''We slice all test files and run through the classifier.  This function
    calls get_mask_times_from_name() and post_process_mask() for each 
    original image name and creates total masks
    return list of [name, outile_mask, outfile_mask_vis]
    '''
    
    names = np.unique(df_pos['name'].values)
    test_list_loc = []
    for i,name in enumerate(names):
        if verbose:
            print ("\nPost-process mask for image:", name)

        mask_norm, mask_raw, overlay_count \
                = post_process_tiles(df_pos, mask_predict_arr, name, 
                       n_classes=n_classes, 
                       super_verbose=super_verbose)

        # define files
        out_file_mask = os.path.join(out_dir, name)
        out_file_vis = os.path.join(out_dir_vis, name)
        out_file_mraw = os.path.join(out_dir_raw, name)
        out_file_count = os.path.join(out_dir_count, name)
        # save to file
        cv2.imwrite(out_file_mask, mask_norm)
        if len(out_dir_vis) > 0:
            tmp_mult = mask_max / np.max(mask_norm)
            cv2.imwrite(out_file_vis, mask_norm * tmp_mult)
        if len(out_dir_raw) > 0:
            tmp_mult = mask_max / np.max(mask_raw)
            cv2.imwrite(out_file_mraw, mask_raw * tmp_mult)
        if len(out_dir_count) > 0:
            tmp_mult = mask_max / np.max(overlay_count)
            cv2.imwrite(out_file_count, overlay_count * tmp_mult)

        # create list
        # the final two columns are for ground truth mask and mask_vis, 
        #   which we aren't tracking here
        outrow = [name, out_file_mask, out_file_vis, '', '']
        test_list_loc.append(outrow)

    return test_list_loc
       
            
###############################################################################
###############################################################################            
def main():
    
    # construct the argument parse and parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='/raid/local/src/siv_net/',
                        help="path to package")
    parser.add_argument('--mode', type=str, default='test',
                        help="test or train")
    parser.add_argument('--model', type=str, default='unet',
                        help="implemented segmentation models [unet]")
    parser.add_argument('--gpu', type=str, default='0',
                        help="GPU number")
    parser.add_argument('--prefix', type=str, default='',
                        help="Naming prefix")
    parser.add_argument('--loss', type=str, default='binary_crossentropy',
                        help="Model loss function [dice, binary_crossentropy]")
    parser.add_argument('--optimizer', type=str, default='SGD',
                        help="Model optimizer [SGD, ADAM, ADAGRAD]")
    parser.add_argument('--epochs', type=int, default=20,
                        help="Training epochs")
    parser.add_argument('--batchsize', type=int, default=32,
                        help="Training epochs")
    parser.add_argument('--model_weights', type=str, default='',
                        help="Weight file")    
    parser.add_argument('--n_bands', type=int, default=3,
                        help="Number of input bands [3, 8]")    
    parser.add_argument('--n_classes', type=int, default=2,
                        help="Number of classes [background, building]")   
    parser.add_argument('--file_list', type=str, default='',
                        help=".txt file holding list of image IDs")
    parser.add_argument('--validation_split', type=float, default=0.1,
                        help="Fraction to hold out of training for validation")   
    parser.add_argument('--early_stopping_patience', type=int, default=4,
                        help="Number of steps to wait for early stopping") 
    parser.add_argument('--slice_x', type=int, default=0,
                        help="Slice images into windows of this width")    
    parser.add_argument('--slice_y', type=int, default=0,
                        help="Slice images into windows of this height")    
    parser.add_argument('--stride_x', type=int, default=0,
                        help="Stride in x direction for training slicing")    
    parser.add_argument('--stride_y', type=int, default=0,
                        help="Stride in y direction for training slicing")    
    parser.add_argument('--resize_height', type=int, default=-1,
                        help="resize height in pixels, set <=0 to use " \
                        + "native height. This is ignored if slicing.")   
    parser.add_argument('--resize_width', type=int, default=-1,
                        help="resize width in pixels, set <=0 to use " \
                        + "native width. This is ignored if slicing.")      
    parser.add_argument('--clip_mask', type=int, default=1,
                        help="switch to plot clip mask values to 1.0")   
    parser.add_argument('--plot_keras_model', type=int, default=0,
                        help="switch to plot the keras model [1 = True]")   
    parser.add_argument('--max_len', type=int, default=1000000,
                        help="Maximum number of train/test images to consider")    
                 
    args = parser.parse_args()
    #######################
    
    print ("\n\nRun basiss.py...")
    print ("args", args)
        
    # Set im_load_resize. Only resize if we're not slicing, and we set 
    #  output_width and output_height > 0
    if (args.slice_x <= 0) and (args.slice_y <= 0) \
                and (args.resize_width > 0) and (args.resize_height > 0):
        im_load_resize = (args.resize_height, args.resize_width)
    else:
        im_load_resize = ()
    print ("im_load_resize:", im_load_resize)
    
    file_list_loc = os.path.join(args.path, 'packaged_data/' + args.file_list)
    
    # model vis
    model_vis_file = os.path.join(args.path, args.model + '.png')

    now = datetime.datetime.now()
    date_string = now.strftime('%Y_%m_%d.%H-%M-%S')
    print ("Date string:", date_string)
    #res_name = args.mode + '__' + args.suffix + '__' + date_string
    
    # define directories
    res0_dir = os.path.join(args.path, 'results')
    res_dir = res0_dir# + res_name
    test_mask_dir = os.path.join(res_dir, args.prefix + '_' + args.mode \
                                 + '_outputs/')
    test_mask_vis_dir = os.path.join(res_dir, args.prefix + '_' \
                        + args.mode + '_outputs_vis/')
    test_count_dir = os.path.join(res_dir, args.prefix + '_' + args.mode \
                        + '_outputs_slice_count/')
    test_raw_dir = os.path.join(res_dir, args.prefix + '_' + args.mode \
                        + '_outputs_slice_raw/')
    outfile_test_locs = os.path.join(args.path, 'packaged_data/' \
                                     + args.prefix + '_' + args.mode \
                                     + '_file_loc.csv')
    
    #res_dir = res0_dir + args.suffix + '/'# + res_name
    for d in [res_dir]:    
        if not os.path.exists(d):
            os.mkdir(d)
    
    model_name_f = os.path.join(res_dir, args.prefix  + '_model_final.hdf5')
    model_name = os.path.join(res_dir, args.prefix  + '_model_best.hdf5')
    model_weights = os.path.join(res_dir, args.model_weights)
    #test_file = res_dir + args.prefix + '_imgs_mask_test.npy'
    shuffle=True
        
    ###########################################################################    
    # setting device here fails, do it in the function call
    #with K.tf.device('/gpu:'+args.gpu):
        #K._set_session(K.tf.Session(config=K.tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)))
    if 2 > 1:   # dummy switch for now, since the above lines often fail 

        # load data
        names, X, X_vis, Y = load_files_universal(file_list_loc, 
                                      nbands=args.n_bands, 
                                      n_classes=args.n_classes,
                                      output_size=im_load_resize,
                                      max_len = args.max_len,
                                      export_im_vis_arr=False,
                                      reshape_mask=False)
        if bool(args.clip_mask):
            Y = np.clip(Y, 0, 1)

        print ("im_arr.shape:", X.shape)
        print ("mask_arr.shape:", Y.shape)
        print ("im_arr.dtype:", X.dtype)
        print ("mask_arr.dtype:", Y.dtype)
        print ("np.max(X)", np.max(X))
        print ("np.max(Y)", np.max(Y))

        if (args.slice_x > 0) and (args.slice_y > 0):
            df_pos_slice, names_slice, X, Y = \
                        slice_ims(X, Y, names, args.slice_x, 
                                  args.slice_y, 
                                  args.stride_x, 
                                  args.stride_y)
            print ("After slicing...")
            print (" im_arr.shape:", X.shape)
            print (" mask_arr.shape:", Y.shape)
            print (" im_arr.dtype:", X.dtype)
            print (" mask_arr.dtype:", Y.dtype)
            print (" np.max(X)", np.max(X))
            print (" np.max(Y)", np.max(Y))
            
        # define algorithm input shape X has shape (N, h, w, nbands)
        input_shape = (X.shape[1], X.shape[2], args.n_bands)
        print ("algo input_shape:", input_shape)



        ####################
        # define model and load data
        if args.model.upper() == 'UNET':
           model = unet(input_shape=input_shape, n_classes=args.n_classes,
                        loss=args.loss, optimizer=args.optimizer)
        else:
            print ("Import segementation model of your choice...")            
            
        ####################
        # Model visualization ?
        if args.plot_keras_model > 0:
            keras.utils.vis_utils.plot_model(model, to_file=model_vis_file, 
                                        show_shapes=True)

        ####################
        # preload weights, if desired
        if len(args.model_weights) > 0:
            t1 = time.time()
            print ("Load weights from:", model_weights)
            model.load_weights(model_weights)   
            print ("Time to load model weights:", time.time() - t1, "seconds")

        ####################
        # set callbacks
        print ("Setting callbacks...")
        model_checkpoint = ModelCheckpoint(model_name, monitor='val_loss', 
                                           save_best_only=True)
        early_stopping = EarlyStopping(monitor='val_loss', 
                                       patience=args.early_stopping_patience, 
                                       verbose=0, mode='auto')
        print ("Callbacks successfully set")

        ####################            
        # set gpu
        os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu) 
    
        print ("\nCheck GPUs:")
        from tensorflow.python.client import device_lib
        print(device_lib.list_local_devices())

        ####################        
        # Train
        if args.mode.upper() == 'TRAIN':    
        
            print ('model_name_f', model_name_f)
            print ('model_name', model_name)
            print ("Start training...")
            t3 = time.time()
            model.fit(X, Y, batch_size=args.batchsize, 
                      epochs=args.epochs,
                      verbose=1, shuffle=shuffle, 
                      validation_split=args.validation_split,
                      callbacks=[model_checkpoint, early_stopping])
            
            print ("Time to fit model:", time.time() - t3, "seconds")
            model.save(model_name_f)
            
        ####################
        # Test
        elif args.mode.upper() == 'TEST':
            # make dirs
            for d in [res_dir, test_mask_dir, test_mask_vis_dir]:    
                if not os.path.exists(d):
                    os.mkdir(d)
                 
            t3 = time.time()
            imgs_mask_test = model.predict(X, batch_size=args.batchsize,
                                           verbose=1)
            print ("Time to predict:", time.time() - t3, "seconds")
            print ("imgs_mask_test.shape", imgs_mask_test.shape)
            print ("imgs_mask_test[0].shape", imgs_mask_test[0].shape)
                
            scores = model.evaluate(X, Y, batch_size=args.batchsize,
                                    verbose=1)
            for i,n in enumerate(model.metrics_names):
                print("%s: %.2f%%" % (model.metrics_names[i], scores[i]*100))
            print ("scores", scores)
            
            ########################
            # POST PROCESS
            # slice, if desired
            if (args.slice_x > 0) and (args.slice_y > 0):
                # make slice dirs
                for d in [test_count_dir, test_raw_dir]:
                    if not os.path.exists(d):
                        os.mkdir(d)

                test_list_loc = post_process_tiles_all(df_pos_slice, 
                                       imgs_mask_test, 
                                       out_dir = test_mask_dir,
                                       out_dir_vis=test_mask_vis_dir,
                                       out_dir_raw=test_raw_dir, 
                                       out_dir_count=test_count_dir, 
                                       n_classes=args.n_classes, 
                                       mask_max=255., 
                                       verbose=True)

            else:
                # Iterate through results and save to file, save in correct
                # format for load_files_universal, with im_test_file as the 
                # output mask and mask_file as the ground truth mask
                test_list_loc = []
                file_list = pd.read_csv(file_list_loc).values
    
                for i,(mask, row) in enumerate(zip(imgs_mask_test, file_list)):
                    [im_test_root, im_test_file, im_vis_file, mask_file, 
                                                         mask_vis_file] = row
                    
                    # put into one grayscale image, discard background layer
                    if args.n_classes == 2:
                        mask2 = mask[:,:,1] 
                    elif args.n_classes == 1:
                        mask2 = mask
                    else:
                        print ("Need to write more code for n_classes > 2")
                        return
                    
                    print ("i", i, "name:", im_test_root, 
                                                   "shape:", mask2.shape)
    
                    mask2_vis = 255. * mask2
                    mask2_vis = mask2_vis.astype(int)
                    # save mask, mask_vis
                    # use complex name with model details
                    #outname = args.prefix + '_' + args.mode + '_im_' \
                    #            + im_test_root.split('.')[0] + '.png'
                    # just use simple name
                    outname = im_test_root.split('.')[0] + '.png'
                    outfile_mask = os.path.join(test_mask_dir, outname)
                    outfile_mask_vis = os.path.join(test_mask_vis_dir, outname)
                    cv2.imwrite(outfile_mask, mask2)        
                    cv2.imwrite(outfile_mask_vis, mask2_vis)   
                    # create list
                    outrow = [im_test_root, outfile_mask, outfile_mask_vis, 
                              mask_file, mask_vis_file]
                    test_list_loc.append(outrow)
                
            # save test_list_loc to file
            header = ['name', 'im_file', 'im_vis_file', 'mask_file', 
                      'mask_vis_file']
            df = pd.DataFrame(test_list_loc, columns=header)
            df.to_csv(outfile_test_locs, index=False)      

            
###############################################################################
###############################################################################
                
if __name__ == '__main__':
    main()
    
    
'''
scp -r /Users/avanetten/Documents/cosmiq/basiss/src 10.123.0.100:/raid/local/src/basiss/

# copy back
scp -r 10.123.0.100://raid/local/src/basiss/results/vegas_400m_unet_3band_8bit_1.5m_slice400_aug_test* /Users/avanetten/Documents/cosmiq/basiss/results
vegas_400m_unet_3band_8bit_1.5m_slice400_aug
'''