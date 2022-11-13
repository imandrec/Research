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

#Import RMSprop from keras

from tensorflow.keras.optimizers import RMSprop

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(learning_rate=0.001),
              metrics=['accuracy'])

#Image generator with database

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1/255)
validation_datagen = ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory(
        './novirus-or-virus/', 
        target_size=(500, 500),
        batch_size=32,
        class_mode='binary') 

validation_generator = validation_datagen.flow_from_directory(
        './validation-novirus-or-virus/', 
        target_size=(500, 500), 
        batch_size=16,
        class_mode='binary')

history = model.fit(
      train_generator,
      steps_per_epoch=3,  
      epochs=5,
      verbose=1,
      validation_data = validation_generator,
      validation_steps=4)

model_json = model.to_json()
with open("model.json","w") as json_file:
  json_file.write(model_json)
  model.save_weights("m.h5")
    
#Run model

import numpy as np
from google.colab import files
from keras.preprocessing import image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


uploaded = files.upload()

for fn in uploaded.keys():
 
  # predicting images
  path = '/content/' + fn
  img = tf.keras.utils.load_img(path, target_size=(500, 500))
  x = tf.keras.utils.img_to_array(img)
  x /= 50
  x = np.expand_dims(x, axis=0)

  images = np.vstack([x])
  classes = model.predict(images, batch_size=32)
  print(classes[0])
    
  if classes[0]>0.08:
    img = mpimg.imread(path)
    imgplot = plt.imshow(img)
    plt.show()

    print("the file" + fn + " contains virus")
    print()
  else:
    img = mpimg.imread(path)
    imgplot = plt.imshow(img)
    plt.show()

    print("the file" + fn + " does not contains virus")
    print()

