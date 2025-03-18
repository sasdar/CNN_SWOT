#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 16:21:32 2024

@author: sarah
"""

#====================================================================================
import numpy as np
import xarray as xr
from glob import glob
import os

#os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
#os.environ['KMP_DUPLICATE_LIB_OK']='True' ##workaround

import tensorflow as tf
from tensorflow.experimental import numpy as tnp
from tensorflow.keras.layers import Layer
#from keras.models import Sequential
from keras.layers import Dense, Conv2D, Input, MaxPooling2D, Dropout, Concatenate, Conv2DTranspose
#from keras import backend as K
from tensorflow.keras.callbacks import TensorBoard

from keras.models import Model, load_model
from keras.callbacks import EarlyStopping,ModelCheckpoint
#from tensorflow.keras.optimizers import *
#from tensorflow.keras import regularizers
from tensorflow.keras.utils import plot_model

import matplotlib
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from netCDF4 import Dataset
import time

import sys

import warnings
warnings.filterwarnings(action='ignore', message='invalid value encountered in cast')

tf.compat.v1.disable_eager_execution()
#tf.compat.v1.enable_eager_execution()
#====================================================================================

#####################################
# Customized Loss Function
#####################################
def custom_loss_function(uSVP,vSVP,angle_swot,lat_swot,svp_tile,ssha_min,ssha_max,uSVP_min,uSVP_max,vSVP_min,vSVP_max,lambda_mean=0.01):
    @tf.function
    def loss(y_true,y_pred):  
        # Define constant
        g = tf.constant(9.81)  # Gravity (m/s^2)
        omega = tf.constant(7.2921e-5)  # Earth's angular velocity (rad/s)
        dx = tf.constant(2e3) # in meters
        dy = tf.constant(2e3)
        dssh_dx = tf.zeros_like(y_pred)
        dssh_dy = tf.zeros_like(y_pred)        

        # Denormalization
        #y_true_d = (y_true * (ssha_max - ssha_min)) + ssha_min
        y_pred_d = tf.math.add(tf.math.multiply(y_pred, tf.math.subtract(ssha_max,ssha_min)),ssha_min)   
        uSVP_d   = tf.math.add(tf.math.multiply(uSVP, tf.math.subtract(uSVP_max,uSVP_min)),uSVP_min)
        vSVP_d   = tf.math.add(tf.math.multiply(vSVP, tf.math.subtract(vSVP_max,vSVP_min)),vSVP_min)
        lat_d    = (lat_swot * 180) -90
                     
        # Compute gradients along the x-axis and y_axis
        dssh_dx = tf.math.divide((tf.math.subtract(y_pred_d[:,:,2::,:],y_pred_d[:,:,0:-2,:])),(2.0*dx))
        dssh_dy = tf.math.divide((tf.math.subtract(y_pred_d[:,2::,:,:],y_pred_d[:,0:-2,:,:])),(2.0*dy))
        
        #tf.print('y_pred :',tf.math.reduce_max(tf.math.abs(y_pred_d)),output_stream=sys.stderr)   
                        
        # Add a first and alast column of 0
        #dssh_dx = tf.concat([tf.zeros_like(dssh_dx_tmp[:, :, :1, :]),dssh_dx_tmp],axis=2)
        #dssh_dx = tf.concat([dssh_dx,tf.zeros_like(dssh_dx_tmp[:, :, -1:, :])],axis=2)
        
        # Add a first and a last row of 0
        #dssh_dy = tf.concat([tf.zeros_like(dssh_dy_tmp[:, :1, :, :]),dssh_dy_tmp],axis=1)
        #dssh_dy = tf.concat([dssh_dy,tf.zeros_like(dssh_dy_tmp[:, -1:, :, :])],axis=1)
        
        #Calculate the Coriolis parameter (f) at the given latitude
        #f = 2.0 * tf.math.multiply(omega,tf.sin((tf.math.multiply(lat_swot,tf.constant(np.pi/180., dtype=tf.float64)))))
        f = 2.0 * omega * tf.sin(lat_d[:,1:-1,1:-1,:] * np.pi / 180.0)
        tf.print('f: ', tf.math.reduce_max(f),output_stream=sys.stderr)
            
        # Calculate the geostrophic velocity components
        ug_mup = tf.math.multiply(tf.math.divide(-g,f),dssh_dy[:,:,1:-1,:])
        vg_mup = tf.math.multiply(tf.math.divide(g,f),dssh_dx[:,1:-1,:,:])
        
        #tf.print('ug_mup:',tf.math.reduce_max(tf.math.abs(tf.math.multiply(ug_mup,svp_tile[:,1:-1,1:-1,:]))),output_stream=sys.stderr)
        #tf.print('vg_mup:',tf.math.reduce_max(tf.math.abs(tf.math.multiply(vg_mup,svp_tile[:,1:-1,1:-1,:]))),output_stream=sys.stderr)
       
        # Rotation
        alpha = angle_swot[:,1:-1,1:-1,:] - 90 # rotation angle
        ug_mup_p = tf.math.add((tf.math.multiply(ug_mup,tf.cos(tf.math.multiply(alpha,tf.constant(np.pi/180.))))), 
                               (tf.math.multiply(vg_mup,tf.sin(tf.math.multiply(alpha,tf.constant(np.pi/180.))))))
        vg_mup_p = tf.math.add((tf.math.multiply(tf.math.negative(ug_mup),tf.sin(tf.math.multiply(alpha,tf.constant(np.pi/180.))))),
                               (tf.math.multiply(vg_mup,tf.cos(tf.math.multiply(alpha,tf.constant(np.pi/180.))))))

        #tf.print('mean ug_mup_p:', tf.reduce_mean(ug_mup_p),output_stream=sys.stderr)
        #tf.print('mean vg_mup_p:', tf.reduce_mean(vg_mup_p),output_stream=sys.stderr)

        uSVP_tile = tf.math.add(tf.zeros_like(ug_mup_p),uSVP_d[:, tf.newaxis, tf.newaxis, :])
        vSVP_tile = tf.math.add(tf.zeros_like(ug_mup_p),vSVP_d[:, tf.newaxis, tf.newaxis, :])
        #print('mean uSVP_tile:', tf.reduce_mean(uSVP_tile))
        #print('mean vSVP_tile: ', tf.reduce_mean(vSVP_tile))

        SE_u = tf.square(tf.math.multiply(tf.math.subtract(uSVP_tile,ug_mup_p),svp_tile[:,1:-1,1:-1,:]))
        SE_v = tf.square(tf.math.multiply(tf.math.subtract(vSVP_tile,vg_mup_p),svp_tile[:,1:-1,1:-1,:]))
                        
        mse_u = tf.reduce_mean(SE_u)
        mse_v = tf.reduce_mean(SE_v)
        
        # Calculate the mean of y_pred_d
        mean_y_pred = tf.reduce_mean(y_pred_d)
        
        # Quadratic penalty: penalty grows more rapidly as the mean moves away from 0
        mean_penalty = tf.square(mean_y_pred)
        
        # Combine the MSE with the mean constraint
        total_loss = mse_u + mse_v + lambda_mean * mean_penalty
        
        return total_loss
    return loss