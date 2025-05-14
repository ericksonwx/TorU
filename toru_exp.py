# U-Net for Tornado Prediction
# Author: Nathan Erickson
# December 1st, 2024

import os
import glob
import time
import pickle
import argparse
import random
import numpy as np
import xarray as xr
import tensorflow as tf
import matplotlib.pyplot as plt
import visualkeras

from datetime import datetime
from keras_unet_collection import models
from keras.layers import BatchNormalization,Concatenate

from parser import create_parser
from custom_losses import RegressLogLoss_Normal,RegressLogLoss_SinhArcsinh,WeightedMSE,predict_lln,predict_shash
from custom_metrics import mae_from_dist
from toru_utils import GradientMonitor,SavePredictionsCallback 

# See Google sheet for upcoming fixes

class toru():
    def init(self):
        # Configure GPU usage
        visible_devices = tf.config.list_physical_devices('GPU') 
        n_visible_devices = len(visible_devices)
        print('GPUS:', visible_devices)
        if n_visible_devices > 0:
            for device in visible_devices:
                tf.config.experimental.set_memory_growth(device, True)
            print('We have %d GPUs\n'%n_visible_devices)
            if n_visible_devices > 1:
                print('Using multiple GPUs right now, make sure you REALLY want to do this in the future')
        else:
            print('NO GPU')

        # Parse arguments from command line
        parser = create_parser()
        self.args = vars(parser.parse_args())
        print(self.args)
        
        # Debug?
        self.debug = self.args['debug']

        # Set distribution type
        self.llshash = True

        # Initiate relevant model variables and directory structures
        print('Starting TorU...')
        
        self.base_dir = '/ourdisk/hpc/ai2es/nerickson/'
        self.inputs_dir = os.path.join(self.base_dir,'Tor_DL/v2/Inputs') 
        self.labels_dir = os.path.join(self.base_dir,'Tor_DL/v2/Labels')
        self.test_in_dir = os.path.join(self.base_dir,'Tor_DL/TCTs/Inputs')
        self.test_lab_dir = os.path.join(self.base_dir,'Tor_DL/TCTs/Labels')
        
        self.out_dir = os.path.join(self.base_dir,f'Tor_DL/Output/{self.args["model"]}')
        self.img_dir = os.path.join(self.base_dir,f'Tor_DL/Images/{self.args["model"]}')
        # Check if output directories exist; if not, create them
        if not os.path.exists(self.out_dir):
            os.mkdir(self.out_dir)
        if not os.path.exists(self.img_dir):
            os.mkdir(self.img_dir)
        
        pass

    def build_loss_dict(self,weight1,weight2,weight3,thresh1,thresh2,thresh3):
        # Build dictionary of loss functions; from Randy's code

        loss_dict = {}
        #this is parametric regression that assumes normal dist.
        loss_dict['RegressLogLoss_Normal'] = RegressLogLoss_Normal(weights=[1.0,weight1,weight2,weight3],thresh=[thresh1,thresh2,thresh3])
        #this is parametric regression that assumes a SHASH dist. 
        loss_dict['RegressLogLoss_SinhArcsinh'] = RegressLogLoss_SinhArcsinh(weights=[1.0,weight1,weight2,weight3],thresh=[thresh1,thresh2,thresh3])
        return loss_dict

    def build_opt_dict(self,learning_rate):
        # Build dictionary of optimizers; from Randy's code
        opt_dict = {}
        opt_dict['adam'] = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        opt_dict['adagrad'] = tf.keras.optimizers.Adagrad(learning_rate=learning_rate)
        opt_dict['sgd'] = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        opt_dict['rmsprop'] = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        return opt_dict       

    def read(self):
        # Start a timer 
        start_read = time.time()
        
        # Read in data for model (possibly in batches, need to learn how to do that)
        print('Reading data...')       
 
        # Change into inputs directory; read in input data
        os.chdir(self.inputs_dir)
        input_files = glob.glob('*00.nc') # Get only the 2021 files for now to test the model
        input_files = sorted(input_files, key=lambda x: datetime.strptime(x[9:27],'%d-%b-%y_%H:%M:%S'))
        if self.debug:
            input_files = input_files[:100]

        # Rename variables, squeeze variable dimension
        input_data = xr.open_mfdataset(input_files,concat_dim='sample',combine='nested',parallel=False).to_dataarray(name='features').squeeze('variable')
        input_data = input_data.transpose('sample','longitude','latitude','feature')

        # Change into labels directory; read in label data
        os.chdir(self.labels_dir)
        label_files = glob.glob('*00.nc')
        label_files = sorted(label_files, key=lambda x: datetime.strptime(x[7:25],'%d-%b-%y_%H:%M:%S'))
        if self.debug:
            label_files = label_files[:100]

        label_data = xr.open_mfdataset(label_files,concat_dim='sample',combine='nested',parallel=False).to_dataarray(name='labels').squeeze('variable')
        label_data = label_data.transpose('sample','longitude','latitude','feature')
        
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
        os.chdir(self.test_in_dir)
        
        tct_input_files = glob.glob('*.nc')
        tct_input_files = sorted(tct_input_files, key=lambda x: datetime.strptime(x[9:27],'%d-%b-%y_%H:%M:%S'))
        
        tct_in_data = xr.open_mfdataset(tct_input_files,concat_dim='sample',combine='nested',parallel=False).to_dataarray(name='labels').squeeze('variable')
        tct_in_data = tct_in_data.transpose('sample','longitude','latitude','feature')
        
        os.chdir(self.test_lab_dir)
        
        tct_label_files = glob.glob('*.nc')
        tct_label_files = sorted(tct_label_files, key=lambda x: datetime.strptime(x[7:25],'%d-%b-%y_%H:%M:%S'))
        
        tct_label_data = xr.open_mfdataset(tct_label_files,concat_dim='sample',combine='nested',parallel=False).to_dataarray(name='labels').squeeze('variable')
        tct_label_data = tct_label_data.transpose('sample','longitude','latitude','feature')

        # Log timesteps for each case to a file for future reference
        os.chdir(self.out_dir)
        with open(f'{self.args["log"]}.txt','w') as f:
            f.write('-----------------------------------')
            f.write('Writing for new run of model')
            f.write('-----------------------------------')
            for file_idx,file in enumerate(label_files):
                f.write(f'Case number {file_idx} corresponds to {file}')
                f.write('\n')
            f.close()

        with open(f'{self.args["tct_log"]}.txt','w') as f:
            for file_idx,file in enumerate(tct_label_files):
                f.write(f'Case number {file_idx} corresponds to {file}')
                f.write('\n')
            f.close()
        print(f'Wrote output in {os.getcwd()}')
        
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
     
        tct_features = tct_full_data[:,:,:,:25]
        tct_labels = tct_full_data[:,:,:,25:]

        train_norm = (train_features - np.min(train_features)) / (np.max(train_features) - np.min(train_features))
        val_norm = (val_features - np.min(train_features)) / (np.max(train_features) - np.min(train_features))
        test_norm = (test_features - np.min(train_features)) / (np.max(train_features) - np.min(train_features))

        train_features = train_norm
        val_features = val_norm
        test_features = test_norm

        tct_norm = (tct_features - np.min(train_features)) / (np.max(train_features) - np.min(train_features))
        # FIX THIS NORMALIZATION

        # Return np arrays for easy access
        self.features = []
        self.features.append(train_features)
        self.features.append(val_features)
        self.features.append(test_features)

        self.labels = []
        self.labels.append(train_labels)
        self.labels.append(val_labels)
        self.labels.append(test_labels)
 
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

        if self.debug: 
            batch_size = 16 # Set explicitly for debugging
        else:
            batch_size = self.args['batch']

        ds_train = ds_train.batch(batch_size)
        ds_val = ds_val.batch(batch_size)
        ds_test = ds_test.batch(batch_size)
        ds_tct = ds_tct.batch(batch_size)       
        
        # Tell us how long all of this took
        end_read = time.time()
        runtime_seconds = end_read - start_read
        runtime_minutes = runtime_seconds/60

        print(f'Preprocessing time: {runtime_minutes:.2f} minutes')
 
        return ds_train,ds_val,ds_test,ds_tct

    def unet(self,ds_train,ds_val):
        print('Building model...')       
        # Build the model, conduct hyperparameter searches, compile

        # Base UNet
        args = self.args 

        # Encoder parameters
        X = ds_train
        filters = args['filters']
        kernel_size = args['kernel_size']
        stack_down = args['stack']
        activation_down = args['activation_conv']
        pool_opt = args['pool_type'] 
        u_name = 'tor_unet'

        # Decoder parameters 
        stack_up = args['stack']
        activation_up = args['activation_conv'] 
        unpool_opt = args['unpool'] 
        concat = True 
       
        activation_out = args['activation_out']
 
        if self.llshash:
            nlabels = 4
        else:
            nlabels = 1
 
        # UNet 3+ (Manually specified hyperparameters)
        model = models.unet_3plus_2d((32,32,25),
                        n_labels=nlabels*3, 
                        l1=args['l1'],
                        l2=args['l2'],
                        dropout=args['dropout'],
                        filter_num_down=filters,
                        filter_num_skip='auto',
                        filter_num_aggregate='auto',
                        stack_num_down=stack_down,
                        stack_num_up=stack_up,
                        activation=activation_up,
                        output_activation=None,
                        batch_norm=True,
                        pool=pool_opt,
                        unpool=unpool_opt,
                        name=u_name)

        # Compile model
        loss_dict = self.build_loss_dict(3,5,7,37,48,59) # Weights, threshold
        opt_dict = self.build_opt_dict(args['lrate']) #Learning rate
        
        metrics = [mae_from_dist]

        if self.llshash:
        # SHASH
            model.compile(
                loss=loss_dict['RegressLogLoss_SinhArcsinh'],
                optimizer=opt_dict['adam'],
                metrics=metrics
            )
        elif self.lln:
        # Gaussian
            model.compile(
                loss=loss_dict['RegressLogLoss_Normal'],
                optimizer=opt_dict['adam']
            )
        elif self.wmse:
        # Weighted MSE
            model.compile(
                loss=loss_dict['WeightedMSE'],
                optimizer=opt_dict['adam']
            )
        else:
        # MSE
            model.compile(
                loss=tf.keras.losses.MeanSquaredError(),
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
 
        if args['verbose']:
            print(model.summary()) 
        
        # Plot model architecture
        if args['verbose']:
            os.chdir(self.img_dir)
            visualkeras.layered_view(model,to_file='unet_architecture_v2.png',legend=True,type_ignore=[BatchNormalization,Concatenate])   
            print(f'Model architecture diagram saved in {os.getcwd()}') 
 
        return model

    def train(self,model,ds_train,ds_val,ds_test,ds_tct):
        print('Training model...')       

        # Set up callbacks to ensure we're getting the best model
        #es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_mae_from_dist',patience=1000,restore_best_weights=True, mode='min')
        nan_callback = tf.keras.callbacks.TerminateOnNaN()
        
        val_batch = ds_val.take(4)
        save_preds_callback = SavePredictionsCallback(val_batch,save_freq=5)

        if self.debug:
            n_epochs = 50
        else:
            n_epochs = 5000

        # Train the model (this is where the heavy lifting takes place)
        start_training = time.time()
        toru_train = model.fit(ds_train,epochs=n_epochs,validation_data=ds_val,
                               callbacks=[nan_callback,save_preds_callback]) # Can also split train/val here
        end_training = time.time()
        runtime_seconds = end_training - start_training
        runtime_minutes = runtime_seconds/60

        print(f'Training time: {runtime_minutes:.2f} minutes')

        #### Plot training vs. validation loss ####
        os.chdir(self.img_dir)
        trained = toru_train
        
        plt.plot(trained.history['loss'],'-r',label='Training Loss')
        plt.plot(trained.history['val_loss'],'-b',label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('SHASH Loss by Epoch')
        plt.grid()
        plt.legend()
        plt.savefig(f'train_val_loss_{self.args["model"]}.jpg',bbox_inches='tight',dpi=300)     
        plt.clf()       
 
        plt.plot(trained.history['mae_from_dist'],'-r',label='Training MAE')
        plt.plot(trained.history['val_mae_from_dist'],'-b',label='Validation MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.title('MAE by Epoch')
        plt.grid()
        plt.legend()
        plt.savefig(f'train_val_mae_{self.args["model"]}.jpg',bbox_inches='tight',dpi=300)     
        print(f'Loss curves in {self.img_dir }')
 
        # Predict on base dataset
        if self.llshash: # Call to sample from log loss SHASH distribution
            params,outs = predict_shash(model,ds_test)
            print('--------------------------------------')
            print('Shape of output array is;',outs.shape)
            print('Shape of parameter array is;',params.shape)
            print('Mean of output array (10th percentile) is;',np.mean(outs[...,0]))
            print('Mean of output array (25th percentile) is;',np.mean(outs[...,1]))
            print('Mean of output array (50th percentile) is;',np.mean(outs[...,2]))
            print('Mean of output array (75th percentile) is;',np.mean(outs[...,3]))
            print('Mean of output array (90th percentile) is;',np.mean(outs[...,4]))
            print('Standard deviation across cases is:',np.std(outs[...,2],axis=0))
        elif self.lln: # Generate raw predictions
            params,outs = predict_lln(model,ds_test)
            # outs[...,0] = 10th, [...,1] = 25th, [...,2] = 50th, [...,3] = 75th, [...,4] = 90th
            print('--------------------------------------')
            print('--------------------------------------')
            print('Shape of output array is;',outs.shape)
            print('Shape of parameter array is;',params.shape)
            print('Mean of output array (10th percentile) is;',np.mean(outs[...,0]))
            print('Mean of output array (25th percentile) is;',np.mean(outs[...,1]))
            print('Mean of output array (50th percentile) is;',np.mean(outs[...,2]))
            print('Mean of output array (75th percentile) is;',np.mean(outs[...,3]))
            print('Mean of output array (90th percentile) is;',np.mean(outs[...,4]))
            print('Standard deviation across cases is:',np.std(outs[...,2],axis=0))

        toru_cheat = model.predict(ds_val) #??	

        # Predict on TCTs
        toru_tct = model.predict(ds_tct) #??	

        # Quit running if debug
        if self.debug:
            return

        # Output model/history
        os.chdir(self.out_dir)
        model.save(f'{self.args["model"]}.keras')
        with open(f'{self.args["model"]}.pkl','wb') as file:
            pickle.dump(trained.history,file)        
        with open(f'{self.args["pred_file"]}_predict_params.pkl','wb') as file:
            pickle.dump(params,file)
        with open(f'{self.args["pred_file"]}_predict_outs.pkl','wb') as file:
            pickle.dump(outs,file)
        with open(f'{self.args["pred_tct_file"]}.pkl','wb') as file:
            pickle.dump(toru_tct, file)
        print('Results in ',os.getcwd()) 

        return toru_train

    def test(self,trained,ds_test):
        print('Testing model...')       
        # Test the model on the test split
        toru_test = trained.predict(ds_test)
        
        pass
    
    def main(self):
        self.init()
        train,val,test,tct = self.read()
        tor_unet = self.unet(train,val)
        trained = self.train(tor_unet,train,val,test,tct)
        self.test(tor_unet,test)

if __name__ == '__main__':
    toru_model = toru()
    toru_model.main()
