# Developed in google colab

!wget https://www.dropbox.com/s/1to9qvipta38fq2/novirus-or-virus.zip
!wget https://www.dropbox.com/s/yeg2vdso5v844al/validation-novirus-or-virus.zip

# Extract files from ZIP
    
import zipfile

local_zip = './novirus-or-virus.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('./novirus-or-virus')

local_zip = './validation-novirus-or-virus.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('./validation-novirus-or-virus')

zip_ref.close()

# Create model

import tensorflow as tf

model = tf.keras.models.Sequential([
    
    # Input shape is the desired size of the image 50x50 with 3 bytes color
    # First convolution
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(500, 500, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    # Second convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    # Third convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    # Fourth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    # Fifth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    
    # 512 neurons in hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    
    # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('no virus') and 1 for the other ('virus')
    tf.keras.layers.Dense(1, activation='sigmoid')
])

#Print model details

model.summary()
