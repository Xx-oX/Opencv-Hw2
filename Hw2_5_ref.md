# Reference for Hw2_5
## [colab](https://colab.research.google.com/drive/1kMTvp6-GJhA3hs9iVCjxoR8krfzRNsVT)

## Model & Train
```
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import os
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Flatten, Dense, Dropout
from tensorflow.python.keras.applications.resnet50 import ResNet50
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.python.keras.utils import to_categorical

DATASET_PATH  = './kagglecatsanddogs_3367a/PetImages'
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 8
FREEZE_LAYERS = 2
NUM_EPOCHS = 5

train_datagen = ImageDataGenerator(rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2) # set validation split

train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training') # set as training data

validation_generator = train_datagen.flow_from_directory(
    DATASET_PATH, # same directory as training data
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation') # set as validation data
	
	
net = ResNet50(include_top=False, weights='imagenet', input_tensor=None,
               input_shape=(IMAGE_SIZE[0],IMAGE_SIZE[1],3))
x = net.output
x = Flatten()(x)

x = Dropout(0.5)(x)

output_layer = Dense(2, activation='softmax', name='softmax')(x)

model = Model(inputs=net.input, outputs=output_layer)
for layer in model.layers[:FREEZE_LAYERS]:
    layer.trainable = False
for layer in model.layers[FREEZE_LAYERS:]:
    layer.trainable = True

model.compile(optimizer=Adam(lr=1e-5),
                  loss='categorical_crossentropy', metrics=['accuracy'])

print(model.summary())

model.fit_generator(
    train_generator,
    steps_per_epoch = train_generator.samples // BATCH_SIZE,	
    validation_data = validation_generator, 
    validation_steps = validation_generator.samples // BATCH_SIZE,
    epochs = NUM_EPOCHS)	
	
model.save('model-resnet50.h5')
```

## resize [ref](https://github.com/NightKirie/COMPUTER-VISION-AND-DEEP-LEARNING_2020/blob/main/Hw2/Q5.py)

```
def Get_Random_Eraser(pixel_level=False):
	def eraser(input_img):
        if input_img.ndim == 3:
            imgh, imgw, imgl = input_img.shape
        elif input_img.ndim == 2:
            imgh, imgw = input_img.shape

        p = np.random.rand()

        if p > 0.5:
            return input_img

        while True:
            s = np.random.uniform(0.02, 0.4) * imgh * imgw
            r = np.random.uniform(0.3, 1/0.3)
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))
            left = np.random.randint(0, imgw)
            top = np.random.randint(0, imgh)

            if left + w <= imgw and top + h <= imgh:
                break

        if pixel_level:
            if input_img.ndim == 3:
                c = np.random.uniform(0, 255, (h, w, imgl))
            if input_img.ndim == 2:
                c = np.random.uniform(0, 255, (h, w))
        else:
            c = np.random.uniform(0, 255)

        input_img[top:top + h, left:left + w] = c

    return input_img
return eraser
```