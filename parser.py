# Argument parser for TorU experiments

# List of arguments to include:
    # File prefixes
    # Hyperparameters 

import argparse

def create_parser():
    # Argument parser

    # Parse the command-line arguments
    parser = argparse.ArgumentParser(description='TorU', fromfile_prefix_chars='@')


    # High-level info for WandB
    parser.add_argument('--project', type=str, default='', help='WandB project name')

    # High-level commands
    parser.add_argument('--check', action='store_true', help='Check results for completeness')
    parser.add_argument('--nogo', action='store_true', help='Do not perform the experiment')
    parser.add_argument('--force', action='store_true', help='Perform the experiment even if the it was completed previously')

    parser.add_argument('--verbose', '-v', action='count', default=0, help="Verbosity level")

    # CPU/GPU
    parser.add_argument('--cpus_per_task', type=int, default=None, help="Number of threads to consume")

    # High-level experiment configuration
    parser.add_argument('--log',type=str,default='unet_log',help='Log file for the model')
    parser.add_argument('--tct_log',type=str,default='unet_tct_log',help='Log file for the model (TCT cases)')
    parser.add_argument('--model',type=str,default='unet_model',help='Model output file name')
    parser.add_argument('--pred_file',type=str,default='pred_file',help='Predictions file name')
    parser.add_argument('--pred_tct_file',type=str,default='pred_tct_file',help='Predictions file name (TCT cases)')
    parser.add_argument('--debug',action='store_true',help='Debug run; useful for testing new changes to model workflow')

    # Specific experiment configuration
    parser.add_argument('--exp_index', type=int, default=None, help='Experiment index')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')

    # General network parameters
    parser.add_argument('--label_channels', type=int, default=7, help='Number of channels in label images')
    parser.add_argument('--lrate', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--activation_out', type=str, default=None, help='Activation function for output')

    # Convolutional unit parameters    
    parser.add_argument('--filters', nargs='+', type=int, default=[32,64,128], help='Number of convolutional filters per layer') 
    parser.add_argument('--kernel_size', type=int, default=3, help='Size of convolutional kernel in preprocessing layer') 
    parser.add_argument('--stride', type=int, default=2, help='Size of convolutional stride in pooling layer') 
    parser.add_argument('--pool', type=int, default=2, help='Size of pooling in layer') 
    parser.add_argument('--pool_type', type=str, default='max', help='Type of pooling to perform') 
    parser.add_argument('--unpool', type=str, default='bilinear', help='Type of unpooling to perform') 
    parser.add_argument('--stack', type=int, default=2, help='Number of convolutional layers to stack in each block') 
    parser.add_argument('--activation_conv', type=str, default='relu', help='Activation function for convolutional layers')

    # Regularization parameters
    parser.add_argument('--dropout', type=float, default=None, help='Dropout rate')
    parser.add_argument('--spatial_dropout', type=float, default=None, help='Dropout rate for convolutional layers')
    parser.add_argument('--l1', type=float, default=None, help="L1 regularization parameter")
    parser.add_argument('--l2', type=float, default=None, help="L2 regularization parameter")

    # Early stopping
    parser.add_argument('--min_delta', type=float, default=0.0, help="Minimum delta for early termination")
    parser.add_argument('--patience', type=int, default=100, help="Patience for early termination")
    parser.add_argument('--monitor', type=str, default="val_loss", help="Metric to monitor for early termination")

    # Training parameters
    parser.add_argument('--batch', type=int, default=256, help="Training set batch size")
    parser.add_argument('--prefetch', type=int, default=3, help="Number of batches to prefetch")
    parser.add_argument('--num_parallel_calls', type=int, default=4, help="Number of threads to use during batch construction")
    parser.add_argument('--cache', type=str, default=None, help="Cache (default: none; RAM: specify empty string; else specify file")
    parser.add_argument('--shuffle', type=int, default=0, help="Size of the shuffle buffer (0 = no shuffle")
    
    parser.add_argument('--repeat', action='store_true', help='Continually repeat training set')
    parser.add_argument('--steps_per_epoch', type=int, default=None, help="Number of training batches per epoch (must use --repeat if you are using this)")

    return parser

