# https://medium.com/datadriveninvestor/generative-adversarial-network-gan-using-keras-ce1c05cfdfd3

import numpy as np
import pickle
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Input
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras.models import model_from_json

from dimensions import DIMENSIONS as IMG_SIZE

print("Running: generator_cnn.py on at "+str(IMG_SIZE)+"x"+str(IMG_SIZE))

# Load classifier (discriminant) # https://machinelearningmastery.com/save-load-keras-deep-learning-models/
json_file = open('classify_cnn.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
classifier = model_from_json(loaded_model_json)
classifier.load_weights("classify_cnn.h5")

# # Opening the noisy photos
# images_pickle = pickle.load(open("noise.pickle", "rb"))

# # Reshape the images.
# images = np.expand_dims(images_pickle, axis=3)

# Build the model.
generator = Sequential([
   Dense(units=256,input_dim=100),
   LeakyReLU(0.2),
   Dense(units=512),
   LeakyReLU(0.2),
   Dense(units=1024),
   LeakyReLU(0.2),
   Dense(units=784, activation="tanh"),
])

# Compile the model.
generator.compile(
  loss="binary_crossentropy",
  optimizer="adam"#lambda: Adam(lr=0.0002, beta_1=0.5)
)

classifier.trainable=False
gan_input = Input(shape=(IMG_SIZE,IMG_SIZE,))
x = generator(gan_input)
gan_output = classifier(x)
gan = Model(inputs=gan_input, outputs=gan_output)
gan.compile(loss='binary_crossentropy', optimizer='adam')
