import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

## Train 
def train():
  import pathlib
  print("Download started...")
  dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
  data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
  data_dir = pathlib.Path(data_dir)

  ## Create a dataset
  batch_size = 32
  img_height = 180
  img_width = 180

  print("Download finished...")

  train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

  val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

  class_names = train_ds.class_names

  print(class_names)
  num_classes = len(class_names)

  ## Standardize the data
  normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)

  normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
  image_batch, labels_batch = next(iter(normalized_ds))
  first_image = image_batch[0]
  # Notice the pixel values are now in `[0,1]`.
  print(np.min(first_image), np.max(first_image)) 

  model = Sequential([
    layers.Input(shape=(img_height, img_width, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
  ])

  model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

  model.summary()

  print("Train started...")
  epochs = 15
  history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
  )
  print("Train finished...")

  model.save('model/')

## Predict
def predict(img_url):

  from os.path import exists
  file_exists = exists('model/saved_model.pb')

  if not file_exists:
    train()

  ## Predict on new data
  sunflower_path = tf.keras.utils.get_file('flower', origin=img_url)

  img_height = 180
  img_width = 180

  img = keras.preprocessing.image.load_img(
      sunflower_path, target_size=(img_height, img_width)
  )
  img_array = keras.preprocessing.image.img_to_array(img)
  img_array = tf.expand_dims(img_array, 0) # Create a batch

  model = tf.keras.models.load_model('model/')
  predictions = model.predict(img_array)
  score = tf.nn.softmax(predictions[0])

  class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
  return class_names[np.argmax(score)]