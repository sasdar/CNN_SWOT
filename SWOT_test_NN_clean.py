#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 15:36:39 2025

@author: Sarah Asdar
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
from keras.layers import Dense, Conv2D, Input, MaxPooling2D, Dropout, Concatenate, Conv2DTranspose, Multiply, MultiHeadAttention
from keras import backend as K
from tensorflow.keras.callbacks import TensorBoard
from keras.layers.core import Lambda

from keras.models import Model, load_model
from keras.callbacks import EarlyStopping,ModelCheckpoint
from tensorflow.keras.utils import plot_model

import matplotlib
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import time

import sys

import warnings
warnings.filterwarnings(action='ignore', message='invalid value encountered in cast')

sys.path.append('/Users/sarah/Documents/OneDrive - CNR/SWOT/scripts')
from custom_loss_function import custom_loss_function

tf.compat.v1.disable_eager_execution()
#====================================================================================

##################################################################
# -------------- begin : user defined variables ---------------- #
##################################################################
path_mup = "/Users/sarah/Documents/WORK/drifter_processing/mup_files/"
#file_mup = path_mup+"mup_data_SVP_SWOT_DUACS_202307_202404_tiles_big_edge_v4_OSTIA_WIND_karin_karin2.nc"
file_mup = path_mup+"mup_data_SVP_SWOT_DUACS_202307_202404_tiles_big_edge_v4_OSTIA_WIND_SWOTL3_karin2.nc"


# ------ NN model configuration parameters ------ #
pat = 50
n_epochs = 1000
model_name = "CNN_SWOT_model_with_DUACS+WIND_10_tiles_new_mup_errDuacs+WIND_attention_vnew_karin2_corr_usvp_corr_wind_2"


# -------------- CNN Input Section  -------------- #

# Choose input variables for the CNN
CNN_INPUTS = {
    'ssha_swot': True,           # SWOT SSH anomaly
    'uSVP': True,                # uSVP velocity
    'vSVP': True,                # vSVP velocity
    'angle_swot': True,          # Cross-track angle
    'lat_swot': True,            # Latitude
    'lon_swot': False,           # Longitude 
    'svp_tile_swot': True,       # Tile position
    'ssha_duacs': True,          # DUACS SSH anomaly
    'err_ssha_duacs': True,      # DUACS SSH error
    'wind': False,               # Wind magnitude
    'u_wind': True,              # u-component of wind
    'v_wind': True,              # v-component of wind
    'sst': False                 # SST
}

# Choose variables for input to the first Conv2D layer
CNN_INPUT_VARIABLES = [
    'ssha_swot',
    'ssha_duacs',
    'u_wind',
    'v_wind',
    #'err_ssha_duacs',
    #'lat_swot' 
]

# Choose variables for the attention layer
ATTENTION_INPUTS = {
    'u_wind': True,
    'v_wind': True,
    'err_ssha_duacs': True,
    'sst': False
}

##################################################################
# ------------- end : user defined variables ---------------- #
##################################################################

####################################
# Read Matchup File
####################################
# Load matchup data
ds_mup_ = xr.open_dataset(file_mup)

# Filter data (exclude equatorial band and high-velocity points)
ds_mup = ds_mup_.where(
    ((ds_mup_.lat_svp_mup > 11) | (ds_mup_.lat_svp_mup < -11)) & 
    ((np.abs(ds_mup_.uSVP_mup) < 1) & (np.abs(ds_mup_.vSVP_mup) < 1)),
    drop=True
)

# Extract variables
variables = {
    'ssha_swot': ds_mup.ssha_swot_mup2.values,
    'angle_swot': ds_mup.angle_swot_mup.values,
    'lat_swot': ds_mup.lat_swot_mup.values,
    'lon_swot': ds_mup.lon_swot_mup.values,
    'svp_tile': ds_mup.svp_tile_mup.values,
    'uSVP': ds_mup.uSVP_mup.values,
    'vSVP': ds_mup.vSVP_mup.values,
    'ssha_duacs': ds_mup.ssha_duacs_mup.values,
    'err_ssha_duacs': ds_mup.err_ssha_duacs_mup.values,
    'ug_duacs': ds_mup.ug_duacs_mup.values,
    'vg_duacs': ds_mup.vg_duacs_mup.values,
    'u_wind': ds_mup.u_wind_mup.values,
    'v_wind': ds_mup.v_wind_mup.values,
    'sst_ostia': ds_mup.sst_ostia_mup.values,
    'ug_swotL3': ds_mup.ug_swotL3_mup.values,
    'vg_swotL3': ds_mup.vg_swotL3_mup.values
}

# Rotate wind components
alpha = variables['angle_swot'] - 90
u_wind_proj = variables['u_wind'] * np.cos(np.deg2rad(alpha)) - variables['v_wind'] * np.sin(np.deg2rad(alpha))
v_wind_proj = variables['u_wind'] * np.sin(np.deg2rad(alpha)) + variables['v_wind'] * np.cos(np.deg2rad(alpha))
wind = np.sqrt(u_wind_proj**2 + v_wind_proj**2)

###################################################
# Remove mean for ssha_swot and ssha_duacs
###################################################
ssha_swot_mean = np.nanmean(variables['ssha_swot'], axis=(1, 2), keepdims=True)
ssha_swot_nomean = variables['ssha_swot'] - ssha_swot_mean

ssha_duacs_mean = np.nanmean(variables['ssha_duacs'], axis=(1, 2), keepdims=True)
ssha_duacs_nomean = variables['ssha_duacs'] - ssha_duacs_mean


###################################################
# Needed in the Customed loss Function
###################################################
ssha_max = np.nanmax(ssha_swot_nomean)
ssha_min = np.nanmin(ssha_swot_nomean)
uSVP_max = np.nanmax(variables['uSVP'])
uSVP_min = np.nanmin(variables['uSVP'])
vSVP_max = np.nanmax(variables['vSVP'])
vSVP_min = np.nanmin(variables['vSVP'])

####################################
#  Normalization
####################################
# Function to normalize data
def normalize_data(data):
    data_min = np.nanmin(data)
    data_max = np.nanmax(data)
    data_norm = (data - data_min) / (data_max - data_min)
    return np.nan_to_num(data_norm, nan=0.0)

# Normalize latitude
lat_norm = (variables['lat_swot'] + 90) / 180
lat_norm = np.nan_to_num(lat_norm, nan=0.0)

# Normalize all variables
normalized_vars = {
    'ssha_swot': normalize_data(ssha_swot_nomean),    
    'uSVP': normalize_data(variables['uSVP']),
    'vSVP': normalize_data(variables['vSVP']),
    'angle_swot': np.nan_to_num(variables['angle_swot'], nan=0.0),
    'lat_swot': lat_norm,
    'lon_swot': variables['lon_swot'],
    'svp_tile_swot' : variables['svp_tile'],
    'ssha_duacs': normalize_data(ssha_duacs_nomean),
    'err_ssha_duacs': normalize_data(variables['err_ssha_duacs']),
    'wind': normalize_data(wind),
    'u_wind': normalize_data(u_wind_proj),
    'v_wind': normalize_data(v_wind_proj),
    'sst_ostia': normalize_data(variables['sst_ostia']),
    'ug_duacs': np.nan_to_num(variables['ug_duacs'],nan=0.0),
    'vg_duacs': np.nan_to_num(variables['vg_duacs'],nan=0.0),
    'ug_swotL3': variables['ug_swotL3'],
    'vg_swotL3': variables['vg_swotL3'],
    'ssha_swot_mean': np.nan_to_num(ssha_swot_mean, nan=0.0),
    'ssha_duacs_mean': np.nan_to_num(ssha_duacs_mean, nan=0.0),
}


#################################################
# Split data into Train/Validation/Test datasets
#################################################
def split_data(data, train_idx, val_idx, test_idx):
    return data[train_idx], data[val_idx], data[test_idx]

n_samples = normalized_vars['ssha_swot'].shape[0]
indices = np.arange(n_samples)
num_blocks = n_samples // 10
blocks = np.array_split(indices, num_blocks)

# ~~~ For reproducibility ~~~
np.random.seed(42)
np.random.shuffle(blocks)

# ~ 80% for training & 20% for test
n_test_blocks = int(num_blocks * 0.20)
n_train_blocks = num_blocks - n_test_blocks
train_blocks = blocks[:n_train_blocks]
test_blocks = blocks[n_train_blocks:]
test_idx = np.concatenate(test_blocks)

# ~ 15% of the training data are kept for validation
n_val_blocks = int(0.15 * n_train_blocks)
n_new_train_blocks = n_train_blocks - n_val_blocks

val_blocks = train_blocks[:n_val_blocks]
new_train_blocks = train_blocks[n_val_blocks:]
val_idx = np.concatenate(val_blocks)
train_idx = np.concatenate(new_train_blocks)

# Split all variables
split_data_dict = {}
for key, data in normalized_vars.items():
    split_data_dict[key] = split_data(data, train_idx, val_idx, test_idx)

"""
# Split longitude, angle data, tile and means
lon_swot_split   = split_data(variables['lon_swot'], train_idx, val_idx, test_idx)
angle_swot_split = split_data(np.nan_to_num(variables['angle_swot'], nan=0.0), train_idx, val_idx, test_idx)
svp_tile_swot_split = split_data(variables['svp_tile'], train_idx, val_idx, test_idx)
ssha_swot_mean_split   = split_data(np.nan_to_num(ssha_swot_mean, nan=0.0), train_idx, val_idx, test_idx)
ssha_duacs_mean_split  = split_data(np.nan_to_num(ssha_duacs_mean, nan=0.0), train_idx, val_idx, test_idx)

# Add longitude and angle data to split_data
split_data_dict['lon_swot']      = lon_swot_split
split_data_dict['angle_swot']    = angle_swot_split
split_data_dict['svp_tile_swot'] = svp_tile_swot_split
split_data_dict['ssha_swot_mean']  = ssha_swot_mean_split
split_data_dict['ssha_duacs_mean'] = ssha_duacs_mean_split
"""

#################################################
# Unpack to create 3 datasets: TRAIN, VAL TEST
# and Reshape data for CNN input
#################################################
# Unpack results
data_train, data_val, data_test = {}, {}, {}

for key, (train, val, test) in split_data_dict.items():
    data_train[key] = np.expand_dims(train, axis=-1)
    data_val[key]   = np.expand_dims(val, axis=-1)
    data_test[key]  = np.expand_dims(test, axis=-1)



##################################################################
#
#                      Model Architecture                        #
#
##################################################################

# Clear session
tf.keras.backend.clear_session()

cnn_train_data = []
cnn_val_data = []
cnn_test_data = []

# ~ Define Inputs
inputs = {}
for var in CNN_INPUTS:
    if CNN_INPUTS[var]:
        inputs[var] = Input(shape=(20, 20, 1) if var not in ['uSVP', 'vSVP'] else (1,), name=var)
        cnn_train_data.append(data_train[var])
        cnn_val_data.append(data_val[var])
        cnn_test_data.append(data_test[var])

# Create `cnn_input` for the first conv layer
cnn_input_list = [inputs[var] for var in CNN_INPUT_VARIABLES if CNN_INPUTS[var]]

# Concatenate selected inputs
cnn_input = Concatenate()(cnn_input_list)

# CNN Layers
conv1 = Conv2D(32, kernel_size=(3, 3), activation='relu', padding="same")(cnn_input)
conv3 = Conv2D(64, kernel_size=(3, 3), activation='relu', padding="same")(conv1)
#conv3 = Conv2D(128, kernel_size=(3, 3), activation='relu', padding="same")(conv2)

# Generalized Attention Layers

attention_outputs = []
for var in ATTENTION_INPUTS:
    if ATTENTION_INPUTS[var]:
        if var in ['u_wind_input', 'v_wind_input']:
            # Special case: Concatenate u_wind and v_wind for the attention layer
            if 'u_wind_input' in ATTENTION_INPUTS and 'v_wind_input' in ATTENTION_INPUTS:
                if ATTENTION_INPUTS['u_wind_input'] and ATTENTION_INPUTS['v_wind_input']:
                    wind_key = Concatenate()([inputs['u_wind_input'], inputs['v_wind_input']])
                    attention_layer = MultiHeadAttention(num_heads=8, key_dim=64)
                    att_output = attention_layer(query=conv3, value=conv3, key=wind_key)
                    attention_outputs.append(att_output)
                    # Skip the individual wind components to avoid duplicate attention layers
                    continue
        # General case: Single variable as key
        attention_layer = MultiHeadAttention(num_heads=8, key_dim=64)
        att_output = attention_layer(query=conv3, value=conv3, key=inputs[var])
        attention_outputs.append(att_output)


# Combine attention outputs with CNN output
if attention_outputs:
    combined_output = Concatenate()([conv3] + attention_outputs)
else:
    combined_output = conv3
    
# Final Layers
conv5 = Conv2D(64, kernel_size=(3, 3), activation='relu', padding="same")(combined_output)
output = Conv2D(1, kernel_size=(1, 1), padding="same")(conv5)

# Define the model
model_inputs = [inputs[var] for var in inputs]
model = Model(inputs=model_inputs, outputs=output)
model.summary()

# Compile the model
optimizer = tf.keras.optimizers.Adam(amsgrad=True, learning_rate=0.001)
model.compile(optimizer=optimizer, loss=custom_loss_function(inputs['uSVP'], inputs['vSVP'], inputs['angle_swot'], inputs['lat_swot'], inputs['svp_tile_swot'],
                                                             ssha_min, ssha_max, uSVP_min, uSVP_max, vSVP_min, vSVP_max))

# Callbacks
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=pat, restore_best_weights=True)
mc = ModelCheckpoint(f'/Users/sarah/Documents/WORK/SWOT/NN_model/{model_name}.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)

# Training
if not os.path.isfile(f'/Users/sarah/Documents/WORK/SWOT/NN_model/{model_name}.h5'):
    ssha_true = np.zeros_like(data_train['ssha_swot'])
    history = model.fit(cnn_train_data, ssha_true, epochs=n_epochs, shuffle=True,
        validation_data=(cnn_val_data, data_val['ssha_swot']),
        callbacks=[es, mc]
    )
    # Plot Loss
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.plot(train_loss, color='blue', label='train')
    plt.plot(val_loss, color='orange', label='validation')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.show()
    
else:
    print(f'!!! {model_name}.h5 ALREADY EXISTS !!!')
    custom_objects = {'loss': custom_loss_function(inputs['uSVP'], inputs['vSVP'], inputs['angle_swot'], inputs['lat_swot'], 
                                                   inputs['svp_tile_swot'],ssha_min, ssha_max, uSVP_min, uSVP_max, vSVP_min, vSVP_max)}
    model = load_model(f'/Users/sarah/Documents/WORK/SWOT/NN_model/{model_name}.h5', custom_objects=custom_objects)



###############################################################
#
#   model validation with independent test data 
#
###############################################################

fitted = model.predict(cnn_train_data)

# Make predictions
test = model.predict(cnn_test_data)

def denormalize_and_add_mean(data, data_min, data_max, mean=None):
    """
    Denormalize data and optionally add mean.
    Args:
        data: Normalized data.
        data_min: Minimum value used for normalization.
        data_max: Maximum value used for normalization.
        mean: Mean value to add (optional).
    Returns:
        Denormalized data.
    """
    denorm_data = data * (data_max - data_min) + data_min
    if mean is not None:
        denorm_data += mean
    return denorm_data


 
# Denormalization SWOT + Add mean for SSH anomaly
ssha_fitted_d  = denormalize_and_add_mean(fitted, ssha_min, ssha_max, data_train['ssha_swot_mean'])
ssha_predict_d = denormalize_and_add_mean(test, ssha_min, ssha_max, data_test['ssha_swot_mean'])

ssha_swot_train_d = denormalize_and_add_mean(data_train['ssha_swot'], ssha_min, ssha_max, data_train['ssha_swot_mean'])
ssha_swot_test_d  = denormalize_and_add_mean(data_test['ssha_swot'], ssha_min, ssha_max, data_test['ssha_swot_mean'])

# Denormalize uSVP and vSVP
uSVP_train_d = denormalize_and_add_mean(data_train['uSVP'], uSVP_min, uSVP_max)
uSVP_test_d = denormalize_and_add_mean(data_test['uSVP'], uSVP_min, uSVP_max)

vSVP_train_d = denormalize_and_add_mean(data_train['vSVP'], vSVP_min, vSVP_max)
vSVP_test_d = denormalize_and_add_mean(data_test['vSVP'], vSVP_min, vSVP_max)

# Denormalize DUACS + Add mean for SSH anomaly
ssha_duacs_train_d = denormalize_and_add_mean(data_train['ssha_duacs'], np.nanmin(variables["ssha_duacs"]), 
                                                   np.nanmax(variables["ssha_duacs"]), data_train['ssha_duacs_mean'])
ssha_duacs_test_d = denormalize_and_add_mean(data_test['ssha_duacs'], np.nanmin(variables["ssha_duacs"]), 
                                                  np.nanmax(variables["ssha_duacs"]), data_test['ssha_duacs_mean'])

# ~ RMSE
rmse_fitted = np.sqrt((np.nansum((ssha_fitted_d.ravel() - ssha_swot_train_d.ravel())**2)) / len(ssha_fitted_d.ravel()))
rmse_test   = np.sqrt((np.nansum((ssha_predict_d.ravel() - ssha_swot_test_d.ravel())**2)) / len(ssha_predict_d.ravel()))

# ~ MAE
mae_fitted = np.abs((np.nansum(ssha_fitted_d.ravel() - ssha_swot_train_d.ravel())) / len(ssha_fitted_d.ravel()))
mae_test   = np.abs((np.nansum(ssha_predict_d.ravel() - ssha_swot_test_d.ravel())) / len(ssha_predict_d.ravel()))

print('RMSE_test: ', rmse_test)
print('MAE_test: ', mae_test)

#ug_duacs_test_d = denormalize_and_add_mean(data_test["ug_duacs"].squeeze(), np.nanmin(variables["ug_duacs"]), np.nanmax(variables["ug_duacs"]))
#vg_duacs_test_d = denormalize_and_add_mean(data_test["vg_duacs"].squeeze(), np.nanmin(variables["vg_duacs"]), np.nanmax(variables["vg_duacs"]))
#ug_duacs_train_d = denormalize_and_add_mean(data_train["ug_duacs"].squeeze(), np.nanmin(variables["ug_duacs"]), np.nanmax(variables["ug_duacs"]))
#vg_duacs_train_d = denormalize_and_add_mean(data_train["vg_duacs"].squeeze(), np.nanmin(variables["vg_duacs"]), np.nanmax(variables["vg_duacs"]))

lat_swot_test_d = (data_test["lat_swot"].squeeze()* 180) - 90
lat_swot_train_d = (data_train["lat_swot"].squeeze() * 180) - 90

####################### SAVE IN FILE ###########################
print('... save in netcdf file ...')
# Define dimensions
dims = {
    'n_tile_test': ssha_swot_test_d.shape[0],
    'n_tile_train': ssha_swot_train_d.shape[0],
    'y_tile': 20,
    'x_tile': 20
}

# Create dataset
ds = xr.Dataset(
    {
        "ssha_predict": (["n_tile_test", "y_tile", "x_tile"], ssha_predict_d.squeeze()),
        "ssha_test": (["n_tile_test", "y_tile", "x_tile"], ssha_swot_test_d.squeeze()),
        "lat_test": (["n_tile_test", "y_tile", "x_tile"], lat_swot_test_d),
        "lon_test": (["n_tile_test", "y_tile", "x_tile"], data_test["lon_swot"].squeeze()),
        "angle_test": (["n_tile_test", "y_tile", "x_tile"], data_test["angle_swot"].squeeze()),
        "tile_test": (["n_tile_test", "y_tile", "x_tile"], data_test["svp_tile_swot"].squeeze()),
        "uSVP_test": (["n_tile_test"], uSVP_test_d.squeeze()),
        "vSVP_test": (["n_tile_test"], vSVP_test_d.squeeze()),
        "ssha_duacs_test": (["n_tile_test", "y_tile", "x_tile"], ssha_duacs_test_d.squeeze()),
        "ug_duacs_test": (["n_tile_test", "y_tile", "x_tile"], data_test["ug_duacs"].squeeze()),
        "vg_duacs_test": (["n_tile_test", "y_tile", "x_tile"], data_test["vg_duacs"].squeeze()),
        "ug_swotL3_test": (["n_tile_test", "y_tile", "x_tile"], data_test["ug_swotL3"].squeeze()),
        "vg_swotL3_test": (["n_tile_test", "y_tile", "x_tile"], data_test["vg_swotL3"].squeeze()),
             
        
        "ssha_fit": (["n_tile_train", "y_tile", "x_tile"], ssha_fitted_d.squeeze()),
        "ssha_train": (["n_tile_train", "y_tile", "x_tile"], ssha_swot_train_d.squeeze()),
        "lat_train": (["n_tile_train", "y_tile", "x_tile"], lat_swot_train_d),
        "lon_train": (["n_tile_train", "y_tile", "x_tile"], data_train["lon_swot"].squeeze()),
        "angle_train": (["n_tile_train", "y_tile", "x_tile"], data_train["angle_swot"].squeeze()),
        "tile_train": (["n_tile_train", "y_tile", "x_tile"], data_train["svp_tile_swot"].squeeze()),
        "uSVP_train": (["n_tile_train"], uSVP_train_d.squeeze()),
        "vSVP_train": (["n_tile_train"], vSVP_train_d.squeeze()),
        "ssha_duacs_train": (["n_tile_train", "y_tile", "x_tile"], ssha_duacs_train_d.squeeze()),
        "ug_duacs_train": (["n_tile_train", "y_tile", "x_tile"], data_train["ug_duacs"].squeeze()),
        "vg_duacs_train": (["n_tile_train", "y_tile", "x_tile"], data_train["vg_duacs"].squeeze()),
        "ug_swotL3_train": (["n_tile_train", "y_tile", "x_tile"], data_train["ug_swotL3"].squeeze()),
        "vg_swotL3_train": (["n_tile_train", "y_tile", "x_tile"], data_train["vg_swotL3"].squeeze()),
    },
    coords={
        "time": np.datetime64("now"),  # Adding timestamp
    },
    attrs={
        "history": f"Created {time.ctime(time.time())}",
    }
)

# Define output path
output_path = "/Users/sarah/Documents/WORK/SWOT/netcdf_NN/" + model_name + ".nc"

# Save to NetCDF
ds.to_netcdf(output_path, format="NETCDF4", engine="netcdf4")

print(f"NetCDF file saved successfully to: {output_path}")






