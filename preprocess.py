import os
import glob
import time
import numpy as np
import xarray as xr
import tensorflow as tf

def pre(inputs_dir,labels_dir,test_in_dir,test_lab_dir):
    # Debug?
    debug = True
        

    # Start a timer 
    start_read = time.time()
    
    # Read in data for model (possibly in batches, need to learn how to do that)
    print('Reading data...')       
    
    # Change into inputs directory; read in input data
    os.chdir(inputs_dir)
    input_files = glob.glob('*.nc') # Get only the 2021 files for now to test the model
    input_files = sorted(input_files, key=lambda x: x[14:16])
    if debug:
        input_files = input_files[:50]
    
    # Rename variables, squeeze variable dimension
    input_data = xr.open_mfdataset(input_files,concat_dim='sample',combine='nested',parallel=False).to_dataarray(name='features').squeeze('variable')
    input_data = input_data.transpose('sample','longitude','latitude','feature')
    
    # Change into labels directory; read in label data
    os.chdir(labels_dir)
    label_files = glob.glob('*.nc')
    label_files = sorted(label_files, key=lambda x: x[14:16])
    if debug:
        label_files = label_files[:50]
    
    label_data = xr.open_mfdataset(label_files,concat_dim='sample',combine='nested',parallel=False).to_dataarray(name='labels').squeeze('variable')
    label_data = label_data.transpose('sample','longitude','latitude','feature')
    #print(label_data)
    
    # Get the number of files in each year
    labels_2020 = [file[14:16] == '20' for file in label_files]
    labels_2021 = [file[14:16] == '21' for file in label_files]
    labels_2022 = [file[14:16] == '22' for file in label_files]
    labels_2023 = [file[14:16] == '23' for file in label_files]
    labels_2024 = [file[14:16] == '24' for file in label_files]
    
    train_len = len(np.where(labels_2020)[0]) + len(np.where(labels_2021)[0]) + len(np.where(labels_2022)[0])
    val_len = len(np.where(labels_2023)[0])
    test_len = len(np.where(labels_2024)[0])
    
    # Change into TCT label directory; read in TCT label data
    os.chdir(test_in_dir)
    
    tct_input_files = glob.glob('*.nc')
    tct_input_files = sorted(tct_input_files, key=lambda x: x[14:16])
    
    tct_in_data = xr.open_mfdataset(tct_input_files,concat_dim='sample',combine='nested',parallel=False).to_dataarray(name='labels').squeeze('variable')
    tct_in_data = tct_in_data.transpose('sample','longitude','latitude','feature')
    
    os.chdir(test_lab_dir)
    
    tct_label_files = glob.glob('*.nc')
    tct_label_files = sorted(tct_label_files, key=lambda x: x[14:16])
    
    tct_label_data = xr.open_mfdataset(tct_label_files,concat_dim='sample',combine='nested',parallel=False).to_dataarray(name='labels').squeeze('variable')
    tct_label_data = tct_label_data.transpose('sample','longitude','latitude','feature')
     
    # Keep all features or just keep ref ones
    ref_data = input_data[:,:,:,:15]
    tct_ref_data = tct_in_data[:,:,:,:15]
    
    azs_data = input_data[:,:,:,15:30]
    tct_azs_data = tct_in_data[:,:,:,15:30]
    
    hrrr_data = input_data[:,:,:,30:]
    tct_hrrr_data = tct_in_data[:,:,:,30:] 
    
    timesteps = np.array([15,22,29])
    label_data = label_data[:,:,:,timesteps] # Only select the reflectivity labels; comment out to keep both ref. and az. shear
    tct_label_data = tct_label_data[:,:,:,timesteps]
    
    input_data = xr.concat([ref_data,hrrr_data], dim='feature')
    
    # Convert input/label datasets to TensorFlow DS
    input_array = input_data.to_numpy()
    label_array = label_data.to_numpy()       
    tct_in_array = tct_in_data.to_numpy()       
    tct_label_array = tct_label_data.to_numpy()       
    
    # Fill missing values in reflectivity labels
    label_array[label_array <= -50] = 0
    tct_label_array[tct_label_array <= -50] = 0 
    
    # Add this back in if only indexing a single timestep
    input_array = np.expand_dims(input_array,axis=-1)
    tct_in_array = np.expand_dims(tct_in_array,axis=-1)
    
    label_array = np.expand_dims(label_array,axis=-1)
    tct_label_array = np.expand_dims(tct_label_array,axis=-1)
    print(np.nanmean(label_array))   
    print(np.nanmean(tct_label_array))   
 
    full_data = np.concatenate([input_array,label_array],axis=3) # Concatenate along 'time' dimension
    tct_full_data = np.concatenate([tct_in_array,tct_label_array],axis=3) # Concatenate along 'time' dimension
    
    # Fill remaining NAs
    full_data = np.nan_to_num(full_data,nan=0)
    tct_full_data = np.nan_to_num(tct_full_data,nan=0)
    
    # Training/testing/validation split
    train_end = val_start = int(len(input_files) * (2/3))
    val_end = test_start = int(len(input_files) * (5/6))
    
    train_data = full_data[:train_end] # Manually setting this split for now, need to fix it later
    val_data = full_data[val_start:val_end]
    test_data = full_data[test_start:]
    
    #train_data = full_data[:train_len] # Split by years
    #val_data = full_data[train_len:val_len]
    #test_data = full_data[val_len:]
    
    # Split into features and labels; normalize features
    train_features,val_features,test_features = train_data[:,:,:,:25],val_data[:,:,:,:25],test_data[:,:,:,:25]
    train_labels,val_labels,test_labels = train_data[:,:,:,25:],val_data[:,:,:,25:],test_data[:,:,:,25:]
    #train_features,val_features,test_features = train_data[:,:,:,:40],val_data[:,:,:,:40],test_data[:,:,:,:40]
    #train_labels,val_labels,test_labels = train_data[:,:,:,40:],val_data[:,:,:,40:],test_data[:,:,:,40:]       
    #print('Test labels:',test_labels)
    
    tct_features = tct_full_data[:,:,:,:25]
    tct_labels = tct_full_data[:,:,:,25:]
    
    train_norm = (train_features - np.min(train_features)) / (np.max(train_features) - np.min(train_features))
    val_norm = (val_features - np.min(train_features)) / (np.max(train_features) - np.min(train_features))
    test_norm = (test_features - np.min(train_features)) / (np.max(train_features) - np.min(train_features))
    
    train_features = train_norm
    val_features = val_norm
    test_features = test_norm
    
    tct_norm = (tct_features - np.mean(train_features)) / np.std(train_features)
    # FIX THIS NORMALIZATION
     
    # Convert to tensors; pad tensors; build TF dataset 
    train_features = tf.convert_to_tensor(train_features,dtype=tf.float32)        
    val_features = tf.convert_to_tensor(val_features,dtype=tf.float32)        
    test_features = tf.convert_to_tensor(test_features,dtype=tf.float32)        
    train_labels = tf.convert_to_tensor(train_labels,dtype=tf.float32)        
    val_labels = tf.convert_to_tensor(val_labels,dtype=tf.float32)        
    
    test_labels = tf.convert_to_tensor(test_labels,dtype=tf.float32)        
    
    tct_features = tf.convert_to_tensor(tct_features,dtype=tf.float32)        
    tct_labels = tf.convert_to_tensor(tct_labels,dtype=tf.float32)        
    
    ds_train = tf.data.Dataset.from_tensor_slices((train_features,train_labels))
    ds_val = tf.data.Dataset.from_tensor_slices((val_features,val_labels))
    ds_test = tf.data.Dataset.from_tensor_slices((test_features,test_labels))
    ds_tct = tf.data.Dataset.from_tensor_slices((tct_features,tct_labels))       
    
    ds_train = ds_train.shuffle(ds_train.cardinality().numpy())
    
    if debug: 
        batch_size = 32
    else:
        batch_size = 256
    
    ds_train = ds_train.batch(batch_size)
    ds_val = ds_val.batch(batch_size)
    ds_test = ds_test.batch(batch_size)
    ds_tct = ds_tct.batch(batch_size)       
    
    # Tell us how long this took
    end_read = time.time()
    runtime_seconds = end_read - start_read
    runtime_minutes = runtime_seconds/60
    
    print(f'Preprocessing time: {runtime_minutes:.2f} minutes')
    
    return ds_train,ds_val,ds_test,ds_tct
