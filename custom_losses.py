import numpy as np
import tensorflow as tf 
import tensorflow.keras.backend as K
from keras import Input,Model
from keras.losses import binary_crossentropy
import tensorflow_probability as tfp


##############################################################################################################
########################################### Classification ###################################################
##############################################################################################################

# Fraction Skill Score (FSS) Loss Function - code taken from: https://github.com/CIRA-ML/custom_loss_functions
# Fraction Skill Score original paper: N.M. Roberts and H.W. Lean, "Scale-Selective Verification of Rainfall
#     Accumulation from High-Resolution Forecasts of Convective Events", Monthly Weather Review, 2008.
def make_fractions_skill_score(mask_size, num_dimensions, c=1.0, cutoff=0.5, want_hard_discretization=False):
    """
    Make fractions skill score loss function. Visit https://github.com/CIRA-ML/custom_loss_functions for documentation.
    Parameters
    ----------
    mask_size: int or tuple
        - Size of the mask/pool in the AveragePooling layers.
    num_dimensions: int
        - Number of dimensions in the mask/pool in the AveragePooling layers.
    c: int or float
        - C parameter in the sigmoid function. This will only be used if 'want_hard_discretization' is False.
    cutoff: float
        - If 'want_hard_discretization' is True, y_true and y_pred will be discretized to only have binary values (0/1)
    want_hard_discretization: bool
        - If True, y_true and y_pred will be discretized to only have binary values (0/1).
        - If False, y_true and y_pred will be discretized using a sigmoid function.
    Returns
    -------
    fractions_skill_score: float
        - Fractions skill score.
    """

    pool_kwargs = {'pool_size': mask_size}
    if num_dimensions == 2:
        pool1 = tf.keras.layers.AveragePooling2D(**pool_kwargs)
        pool2 = tf.keras.layers.AveragePooling2D(**pool_kwargs)
    elif num_dimensions == 3:
        pool1 = tf.keras.layers.AveragePooling3D(**pool_kwargs)
        pool2 = tf.keras.layers.AveragePooling3D(**pool_kwargs)
    else:
        raise ValueError("Number of dimensions can only be 2 or 3")

    @tf.function()
    def fractions_skill_score(y_true, y_pred):
        """ Fractions skill score loss function """
        if want_hard_discretization:
            y_true_binary = tf.where(y_true > cutoff, 1.0, 0.0)
            y_pred_binary = tf.where(y_pred > cutoff, 1.0, 0.0)
        else:
            y_true_binary = tf.math.sigmoid(c * (y_true - cutoff))
            y_pred_binary = tf.math.sigmoid(c * (y_pred - cutoff))

        y_true_density = pool1(y_true_binary)
        n_density_pixels = tf.cast((tf.shape(y_true_density)[1] * tf.shape(y_true_density)[2]), tf.float32)

        y_pred_density = pool2(y_pred_binary)

        MSE_n = tf.keras.metrics.mean_squared_error(y_true_density, y_pred_density)

        O_n_squared_image = tf.keras.layers.Multiply()([y_true_density, y_true_density])
        O_n_squared_vector = tf.keras.layers.Flatten()(O_n_squared_image)
        O_n_squared_sum = tf.reduce_sum(O_n_squared_vector)

        M_n_squared_image = tf.keras.layers.Multiply()([y_pred_density, y_pred_density])
        M_n_squared_vector = tf.keras.layers.Flatten()(M_n_squared_image)
        M_n_squared_sum = tf.reduce_sum(M_n_squared_vector)

        MSE_n_ref = (O_n_squared_sum + M_n_squared_sum) / n_density_pixels

        my_epsilon = tf.keras.backend.epsilon()  # this is 10^(-7)

        if want_hard_discretization:
            if MSE_n_ref == 0:
                return MSE_n
            else:
                return MSE_n / MSE_n_ref
        else:
            return MSE_n / (MSE_n_ref + my_epsilon)

    return fractions_skill_score

def csi(use_as_loss_function, use_soft_discretization,
            hard_discretization_threshold=None):
            
        def loss(target_tensor, prediction_tensor):
            if hard_discretization_threshold is not None:
                prediction_tensor = tf.where(
                    prediction_tensor >= hard_discretization_threshold, 1., 0.
                )
            elif use_soft_discretization:
                prediction_tensor = K.sigmoid(prediction_tensor)
        
            num_true_positives = K.sum(target_tensor * prediction_tensor)
            num_false_positives = K.sum((1 - target_tensor) * prediction_tensor)
            num_false_negatives = K.sum(target_tensor * (1 - prediction_tensor))
            
            denominator = (
                num_true_positives + num_false_positives + num_false_negatives +
                K.epsilon()
            )

            csi_value = num_true_positives / denominator
            
            if use_as_loss_function:
                return 1. - csi_value
            else:
                return csi_value
        
        return loss

class WeightedMSE(tf.keras.losses.Loss):
    """ 
    Calculate a weighted MSE. This loss gives you control to weight the 
    pixels that are > 0 differently than the pixels that are 0 in y_true. This
    class is subclassed from tf.keras.lossess.Loss to hopefully enable 
    'stateful'-ness ?

    weights[0] is the weight for non-zero pixels
    weights[1] is the weight for zero pixels. 

    """
    def __init__(self, weights=[1.0,1.0],name="custom_mse",
                 **kwargs):
        super(WeightedMSE,self).__init__(name=name, **kwargs)

        #store weights
        self.w1 = weights[0]
        self.w2 = weights[1]

    def call(self, y_true, y_pred):

        #build weight_matrix 
        ones_array = tf.ones_like(y_true)
        weights_for_nonzero = tf.math.multiply(ones_array,self.w1)
        weights_for_zero = tf.math.multiply(ones_array,self.w2)
        weight_matrix = tf.where(tf.greater(y_true,1),weights_for_nonzero,
                           weights_for_zero)
        loss = tf.math.reduce_mean(tf.math.multiply(weight_matrix,
                                                    tf.math.square(tf.math.subtract(y_pred,y_true))))
        return loss

class RegressLogLoss_Normal(tf.keras.losses.Loss):
    """Regression log-loss that includes uncertainty via a 2-output regression setup.
    """
    
    def __init__(self,weights=[1.0,1.0],thresh=0.0,low=0,high=75):
        super().__init__()
        #store weights
        self.w1 = weights[0]
        self.w2 = weights[1]
        self.t = thresh
        self.low=low
        self.high=high

    def __str__(self):
        return ("RegressLogLoss_Normal()")

    def call(self, y_true, y_pred): 
        y_true = tf.cast(y_true[...,0], tf.float64) # Changed from y_true[...,0]
        y_pred = tf.cast(y_pred, tf.float64)
        y_pred = tf.squeeze(y_pred)

        root_power = tf.constant(1.,tf.float64)/tf.math.multiply(tf.constant(10.,tf.float64),tf.cast(tf.math.exp(1.),tf.float64))
        
        mu = y_pred[...,0] 
        sigma = tf.math.pow(tf.math.exp(y_pred[...,1]),root_power)

        #tf.print("y_true mean:", tf.reduce_mean(y_true), "std:", tf.math.reduce_std(y_true))
        #tf.print("mu mean:", tf.reduce_mean(mu), "std:", tf.math.reduce_std(mu))
        #tf.print("sigma mean:", tf.reduce_mean(sigma), "std:", tf.math.reduce_std(sigma))
        #tf.print("Sample y_true:", y_true[:5])
        #tf.print("Sample mu:", mu[:5])
        #tf.print("Sample sigma:", sigma[:5])
        
        dist = tfp.distributions.TruncatedNormal(loc=mu,scale=sigma,low=self.low,high=self.high)
 
        loss = -tf.math.log(dist.prob(y_true) + tf.keras.backend.epsilon())

        #build weight_matrix 
        #ones_array = tf.ones_like(y_true)
        #weights_for_nonzero = tf.math.multiply(ones_array,self.w1)
        #weights_for_zero = tf.math.multiply(ones_array,self.w2)
        #weight_matrix = tf.where(tf.greater(y_true,self.t),weights_for_nonzero,
        #                   weights_for_zero)
        
        #weight non-zero pixels more
        #loss = tf.math.multiply(weight_matrix,loss)
    
        # Add lambda diversity penalty
        #lambda_diversity = 0.01
        #batch_mu_std = tf.math.reduce_std(mu, axis=0)
        #batch_sigma_std = tf.math.reduce_std(sigma, axis=0)
        #diversity_loss = lambda_diversity * (1.0 / (batch_mu_std + 1e-6) + 1.0 / (batch_sigma_std + 1e-6))
        #batch_mean = tf.reduce_mean(mu, axis=0)
        #diversity_loss = lambda_diversity * tf.reduce_mean(tf.square(mu - batch_mean))
        #loss += diversity_loss
    
        # Add MSE component of loss
        #lambda_mse = 0.1
        #mse_loss = lambda_mse * tf.reduce_mean(tf.square(mu - y_true))
        #loss += mse_loss

        return tf.reduce_mean(loss) 
      
def predict_lln(model, test):
    # Cast predictions to TF 
    x_true = np.concatenate([x for x,y in test]) #Access each set of features
    y_true = np.concatenate([y for x,y in test]) #Access each set of labels
    y_true = y_true[...,0]
    y_pred = model.predict(x_true)        
    print(test)    
    #print(test.shape)

    #print('Shape of features in loss function is:',x_true.shape)
    #print('Shape of true labels in loss function is:',y_true.shape)       
    #print('Shape of predicted labels in loss function is:',y_pred.shape)       
    #print('-----------------------')
    #print('Case 0:')
    #print(x_true[0,:,:,0])
    #print(y_true[0])
    #print(y_pred[0,:,:,0])
    #print('-----------------------')
    
    #print('Case -1:') 
    #print(x_true[-1,:,:,0])
    #print(y_true[-1])
    #print(y_pred[-1,:,:,0])
    #print('-----------------------')
    y_true = tf.cast(y_true, tf.float64)
    y_pred = tf.cast(y_pred, tf.float64)
    
    #print('Array of mu values (1 case):',y_pred[0,:,:,0])
    #print('Array of sigma values (1 case):',y_pred[0,:,:,1])
    
    y_out10 = np.empty_like(y_true)
    y_out25 = np.empty_like(y_true)
    y_out50 = np.empty_like(y_true)
    y_out75 = np.empty_like(y_true)
    y_out90 = np.empty_like(y_true)
    #print(y_out.shape)
    for i in range(len(y_true)):        
        root_power = tf.constant(1.,tf.float64)/tf.math.multiply(tf.constant(10.,tf.float64),tf.cast(tf.math.exp(1.),tf.float64))
        mu = y_pred[i,:,:,0] 
        sigma = tf.math.pow(tf.math.exp(y_pred[i,:,:,1]),root_power)
        #print(mu)
        #print(sigma)       
    
        dist = tfp.distributions.TruncatedNormal(loc=mu,scale=sigma,low=0,high=75)
        prob_ytrue = dist.prob(y_true)
        #print('Probability of y_true:',prob_ytrue)
        #print('Distribution object:',dist)
    
        y_out10[i,:,:] = dist.quantile(0.1)       
        y_out25[i,:,:] = dist.quantile(0.25)       
        y_out50[i,:,:] = dist.quantile(0.5)       
        y_out75[i,:,:] = dist.quantile(0.75)
        y_out90[i,:,:] = dist.quantile(0.9)
   
    y_out = np.stack([y_out10,y_out25,y_out50,y_out75,y_out90],axis=-1)
    print(y_out.shape)
    return y_pred,y_out
    
class RegressLogLoss_SinhArcsinh(tf.keras.losses.Loss):
    """

    This is the negative log likelihood loss function for regression. 
    More specifically, this is an adaptation from the Barnes technique.

    This has weights that can be applied to prevent under-estimation of rare instance. 

    There is also a threshold value to apply to the weight to. 

    By default these parameters are set to 1 such that no weights are applied.

    """
    def __init__(self,weights=[1.0,1.0,1.0,1.0],thresh=[0.0,0.0,0.0]):
        super().__init__()
        #weights for weighted loss. w1 is > threshold weights. w2 is everything else. 
        self.w0 = weights[0]
        self.w1 = weights[1]
        self.w2 = weights[2]
        self.w3 = weights[3]
        #threshold to apply weights to  
        self.t1 = thresh[0]
        self.t2 = thresh[1]
        self.t3 = thresh[2]
        
        self.t1 = tf.cast(self.t1, tf.float64)
        self.t2 = tf.cast(self.t2, tf.float64)
        self.t3 = tf.cast(self.t3, tf.float64)
        self.w1 = tf.cast(self.w1, tf.float64)
        self.w2 = tf.cast(self.w2, tf.float64)
        self.w3 = tf.cast(self.w3, tf.float64)

    def __str__(self):
        return ("RegressLogLoss_SinhArcsinh()")

    def call(self, y_true, y_pred):

        #force float64 to allow for big and small nums 
        y_true = tf.cast(y_true[...,0], tf.float64) # True label of shape (32,32,15) - Need to calculate mu,sigma,etc. and pass these as part of input tensor
        y_pred = tf.cast(y_pred, tf.float64) # Want predictions of shape (32,32,15,4)
        if 1 < y_true.shape[-1] < 32: # If running with multiple timesteps
            ntimesteps = y_true.shape[-1] 
            y_pred = tf.reshape(y_pred,(tf.shape(y_true)[0],32,32,3,4))
 
        #Chase adaptation to prevent bad inital params 
        root_power = tf.constant(1.,tf.float64)/tf.math.multiply(tf.constant(10.,tf.float64),tf.cast(tf.math.exp(1.),tf.float64))
        
        #these are the 4 parameters of the dist.
        mu = y_pred[...,0] # Pulling these out reduces shapes to (32,32), from (32,32,15)
        sigma = tf.math.pow(tf.math.exp(y_pred[...,1]),root_power)
        gamma = y_pred[...,2] 
        #tau = tf.math.pow(tf.math.exp(y_pred[...,3]),root_power)
        tau = tf.math.softplus(y_pred[...,3]) # Needed since apparently this won't return a positive naturally
 
        #build dists.
        cond_dist = tfp.distributions.SinhArcsinh(mu, sigma,skewness=gamma,tailweight=tau)
 
        #Chase adaptation to prevent HUGE inital losses
        loss = -tf.math.log(cond_dist.prob(y_true)+tf.keras.backend.epsilon())
         
        #Erickson adaptation to enable weighted loss function at multiple thresholds  
        ones_array = tf.ones_like(y_true)
        weights_for_zero = tf.math.multiply(ones_array,self.w0)

        # This is being hardcoded for now, but might be able to do more eloquently
        weights = tf.where(y_true > self.t1, self.w1, weights_for_zero)
        weights = tf.where(y_true > self.t2, self.w2, weights)
        weights = tf.where(y_true > self.t3, self.w3, weights) 
        loss = tf.math.multiply(weights,loss)
         
        # Add MSE component of loss
        #lambda_mse = 0.5
        #mse_loss = lambda_mse * tf.reduce_mean(tf.square(mu - y_true))
        #loss += mse_loss
        
        #return loss 
        return tf.reduce_mean(loss)
     
    def predict(self, model, test):
        # Run model predict - Take test dataset make predictions of y given x
        y_true = np.concatenate([y for x,y in test]) #Access each set of labels
        y_true = y_true[...,0]
        y_pred = model.predict(test)        
        print('Label shape at predict time is:', y_true.shape)
        print('Prediction shape at predict time is:', y_pred.shape)
 
        # Cast predictions to TF 
        y_true = tf.cast(y_true, tf.float64)
        y_pred = tf.cast(y_pred, tf.float64)
        
        root_power = tf.constant(1.,tf.float64)/tf.math.multiply(tf.constant(10.,tf.float64),tf.cast(tf.math.exp(1.),tf.float64))
        
        # Split distribution
        mu = y_pred[...,0]       
        sigma = tf.math.pow(tf.math.exp(y_pred[...,1]),root_power)
        gamma = y_pred[..., 2] 
        tau = tf.math.pow(tf.math.exp(y_pred[...,3]),root_power)

        dist = tfp.distributions.SinhArcsinh(loc=mu, scale=sigma, skewness=gamma, tailweight=tau)

        y_out = dist.quantile(0.5) 
        y_out75 = dist.quantile(0.75)
        y_out90 = dist.quantile(0.9)
        
        return y_out,y_out75,y_out90
       
def predict_shash(model, test):
    # Cast predictions to TF 
    x_true = np.concatenate([x for x,y in test]) #Access each set of features
    y_true = np.concatenate([y for x,y in test]) #Access each set of labels
    y_true = y_true[...,0]
    y_pred = model.predict(x_true)        
   
    y_true = tf.cast(y_true, tf.float64)
    y_pred = tf.cast(y_pred, tf.float64)
    if 1 < y_true.shape[-1] < 32: # If running with multiple time steps
        #ntimesteps = y_true.shape[-1] # Divide length of last dimension by number of parameters
        ntimesteps = 3
        y_pred = tf.reshape(y_pred,(tf.shape(y_pred)[0],32,32,ntimesteps,4))
        print('Reshaped prediction array')
    else:
        print('Passing on without reshaping')

    # Need to reshape here
    
    y_out10 = np.empty_like(y_true)
    y_out25 = np.empty_like(y_true)
    y_out50 = np.empty_like(y_true)
    y_out75 = np.empty_like(y_true)
    y_out90 = np.empty_like(y_true)
    
    root_power = tf.constant(1.,tf.float64)/tf.math.multiply(tf.constant(10.,tf.float64),tf.cast(tf.math.exp(1.),tf.float64))
    
    mu = y_pred[...,0] # Need an extra index here
    sigma = tf.math.pow(tf.math.exp(y_pred[...,1]),root_power)
    gamma = y_pred[...,2] 
    #tau = tf.math.pow(tf.math.exp(y_pred[...,3]),root_power)
    tau = tf.math.softplus(y_pred[...,3])

    dist = tfp.distributions.SinhArcsinh(loc=mu, scale=sigma, skewness=gamma, tailweight=tau)
 
    y_out10 = dist.quantile(0.1)       
    y_out25 = dist.quantile(0.25)       
    y_out50 = dist.quantile(0.5)       
    y_out75 = dist.quantile(0.75)
    y_out90 = dist.quantile(0.9) 

    y_out = np.stack([y_out10,y_out25,y_out50,y_out75,y_out90],axis=-1)
    return y_pred,y_out
