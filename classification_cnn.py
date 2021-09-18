# https://victorzhou.com/blog/keras-cnn-tutorial/
# https://machinelearningmastery.com/save-load-keras-deep-learning-models/

import numpy as np
import pickle
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Activation, Dropout
from keras.utils import to_categorical

from variables import DIMENSIONS, FOLDERS

print("Running: classification_cnn.py on at "+str(DIMENSIONS)+"x"+str(DIMENSIONS))

# Opening the files about data
images_pickle = pickle.load(open("images.pickle", "rb"))
labels_pickle = pickle.load(open("labels.pickle", "rb"))

# Normalize the images.
images_pickle = images_pickle/255.0

# Reshape the images.
images = np.expand_dims(images_pickle, axis=3)

# Add labels.
labels = labels_pickle

from math import log as ln
def roundlog(n): return round(ln(n)/(ln(2)))

# Build the model.
model = Sequential([
  Conv2D(64, kernel_size=roundlog(DIMENSIONS), activation="relu", input_shape=(DIMENSIONS,DIMENSIONS,1)),
  MaxPooling2D(pool_size=(2, 2)),
  Conv2D(64, kernel_size=3, activation="relu"),
  MaxPooling2D(pool_size=(2, 2)),
  Flatten(),
  Dense(len(FOLDERS), activation="softmax"),
])

# model.summary()




# Compile the model.
model.compile(
  optimizer="adam",
  loss="categorical_crossentropy",
  metrics=["accuracy"],
)

# Train the model.

model.fit(
  images,
  to_categorical(labels),
  epochs=5,
  validation_split=0.4,
)

# Save the model to disk.
model_json = model.to_json()
with open("classify_cnn.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("classify_cnn.h5")