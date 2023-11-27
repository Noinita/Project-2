# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 13:33:58 2023

@author: noini
"""
import os

# Set environment variable to avoid OpenMP error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Now, import TensorFlow
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten
from tensorflow.keras.activations import relu, elu, softmax
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
import numpy as np

import numpy
import pandas
import matplotlib.pyplot as plt

#Step 1: Data Processing 

#defining the input image shape in width, height, channel
input_shape = (100,100,3)

#Splitting the Train and Test Data
train_data_dir = 'data/Train'
validation_data_dir = 'data/Validation'
test_data_dir = 'data/Test'

#Data Augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True
)
#Data Augmentation for validation data
validation_datagen = ImageDataGenerator(rescale=1./255)

#Train and Validation generator
batch_size = 32
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical'  
)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical'
)

#Step 2: Neural Network Architecture Design
model = Sequential()

#CV layers
#32 is the number of filters, (3,3) is kernel 
model.add(Conv2D(32, (3, 3), strides=(1, 1), activation='relu', input_shape=input_shape))
model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))

#Pooling Layers
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(MaxPooling2D(pool_size=(2, 2)))

#Flatten Layer
model.add(Flatten())

#Dense layers and Dropout
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))  
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))  
model.add(Dense(4, activation='softmax'))

#Step 3 Hyperparameter Analysis
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['categorical_crossentropy','accuracy'])
model.summary()

#Setting epochs
epochs = 20
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)

# Evaluate the model
result = model.evaluate(validation_generator)
loss = result[0]
accuracy = result[1]
print(f"Validation Accuracy: {accuracy*100:.2f}%")

#Plotting
# Plot training and validation loss
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Model Loss')
plt.legend()

# Plot training and validation accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
plt.legend()

plt.tight_layout()
plt.show()


model.save("Project_2.h5")