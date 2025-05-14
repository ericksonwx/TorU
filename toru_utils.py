import os
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

class GradientMonitor(tf.keras.callbacks.Callback):
    def __init__(self,model,loss_fn,log_freq=1):
        super().__init__()
        self.loss_fn = loss_fn
        self.log_freq = log_freq

    def on_epoch_end(self,epoch,logs=None):
        if epoch % self.log_freq == 0:
            batch = self.validation_data
            x_batch, y_batch = batch[0],batch[1]

            with tf.GradientTape() as tape:
                preds = self.model(x_batch,training=True)
                loss = self.loss_fn(y_batch,preds)

            gradients = tape.gradient(loss,self.model.trainable_variables)

            grad_magnitudes = [
                tf.norm(grad).numpy() if grad is not None else 0
                for grad in gradients
            ]

            mean_grad = np.mean(grad_magnitudes)
            max_grad = np.max(grad_magnitudes)

            print(f'Epoch {epoch}: Mean Gradient Magnitude = {mean_grad}, Max Gradient Magnitude = {max_grad}')

            logs['mean_gradient'] = mean_grad
            logs['max_gradient'] = max_grad

class SavePredictionsCallback(tf.keras.callbacks.Callback):
    def __init__(self, dataset, save_dir="predictions", log_file='log_preds.txt', save_freq=5):
        """
        Custom callback to save model predictions after each epoch.

        Args:
            dataset: A TensorFlow dataset (or NumPy array) to generate predictions on.
            save_dir: Directory where predictions will be saved.
            save_freq: Frequency (every N epochs) to save predictions.
        """
        super().__init__()
        self.dataset = dataset
        self.save_dir = save_dir
        self.log_file = log_file
        self.save_freq = save_freq

        # Ensure save directory exists
        os.makedirs(save_dir, exist_ok=True)

        # Open log file for writing
        self.log_f = open(log_file, 'a')

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.save_freq == 0:
            predictions = []
            labels = []

            for x_batch, y_batch in self.dataset:
                preds = self.model(x_batch, training=False)  # Get predictions
                # Build probability distribution; extract median 

                predictions.append(preds.numpy())  # Convert to NumPy
                labels.append(y_batch.numpy())

            # Convert to single NumPy arrays
            predictions = np.concatenate(predictions, axis=0)
            labels = np.concatenate(labels, axis=0)
            std_across_cases = np.std(predictions, axis=0)
            
            log_msg = (
                f"std_across_cases={np.mean(std_across_cases):.4f}\n"
            )
            print(log_msg)  # Print to console

            ll = True
            # Extract mu and sigma
            if ll == True:
                predictions = tf.cast(predictions,tf.float64)
                root_power = tf.constant(1.,tf.float64)/tf.math.multiply(tf.constant(10.,tf.float64),tf.cast(tf.math.exp(1.),tf.float64))
                mu_pred = predictions[...,0]
                sigma_pred = tf.math.pow(tf.math.exp(predictions[...,1]),root_power)

                #Compute standard deviation across the batch axis
                mu_std_across_cases = np.std(mu_pred, axis=0)
                sigma_std_across_cases = np.std(sigma_pred, axis=0)
                
                #np.save(os.path.join(self.save_dir, f"mu_epoch_{epoch}.npy"), mu_pred)
                #np.save(os.path.join(self.save_dir, f"sigma_epoch_{epoch}.npy"), sigma_pred)
            
                # Compute statistics
                mu_mean, mu_min, mu_max = np.mean(mu_pred), np.min(mu_pred), np.max(mu_pred)
                sigma_mean, sigma_min, sigma_max = np.mean(sigma_pred), np.min(sigma_pred), np.max(sigma_pred)

                # Write stats to log file
                log_msg = (
                    f"Epoch {epoch} - mu: mean={mu_mean:.4f}, min={mu_min:.4f}, max={mu_max:.4f}, std_across_cases={np.mean(mu_std_across_cases):.4f} | "
                    f"sigma: mean={sigma_mean:.4f}, min={sigma_min:.4f}, max={sigma_max:.4f}, std_across_cases={np.mean(sigma_std_across_cases):.4f}\n"
                )
                print(log_msg)  # Print to console
                self.log_f.write(log_msg)  # Write to log file
                self.log_f.flush()  # Ensure data is written

            else:
                np.save(os.path.join(self.save_dir, f'pred_epoch_{epoch}.npy'), predictions)

            print(f"Saved predictions for epoch {epoch}")

    def on_train_end(self, logs=None):
        """Close the log file when training ends."""
        self.log_f.close()
