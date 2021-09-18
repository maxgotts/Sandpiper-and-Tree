import numpy as np
import os
from matplotlib import pyplot as plt
import cv2
import random
import pickle

from dimensions import DIMENSIONS as IMG_SIZE

print("Running: noise_image_process.py at "+str(IMG_SIZE)+"x"+str(IMG_SIZE))

def add_noise(noise_typ,image): # directly from https://stackoverflow.com/questions/22937589/how-to-add-noise-gaussian-salt-and-pepper-etc-to-image-in-python-with-opencv
   if noise_typ == "gauss":
      row,col,ch= image.shape
      mean = 0
      var = 0.1
      sigma = var**0.5
      gauss = np.random.normal(mean,sigma,(row,col,ch))
      gauss = gauss.reshape(row,col,ch)
      noisy = image + gauss
      return noisy
   elif noise_typ == "s&p":
      row,col,ch = image.shape
      s_vs_p = 0.5
      amount = 0.004
      out = np.copy(image)
      # Salt mode
      num_salt = np.ceil(amount * image.size * s_vs_p)
      coords = [np.random.randint(0, i - 1, int(num_salt))
               for i in image.shape]
      out[coords] = 1

      # Pepper mode
      num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
      coords = [np.random.randint(0, i - 1, int(num_pepper))
               for i in image.shape]
      out[coords] = 0
      return out
   elif noise_typ == "poisson":
      vals = len(np.unique(image))
      vals = 2 ** np.ceil(np.log2(vals))
      noisy = np.random.poisson(image * vals) / float(vals)
      return noisy
   elif noise_typ =="speckle":
      row,col,ch = image.shape
      gauss = np.random.randn(row,col,ch)
      gauss = gauss.reshape(row,col,ch)        
      noisy = image + image * gauss
      return noisy

DATADIR = "data/noise"

# Checking for all images in the data folder
for img in os.listdir(DATADIR):
   img_array = cv2.imread(os.path.join(DATADIR, img), cv2.IMREAD_GRAYSCALE)

training_data = []

def create_training_data():
   for img in os.listdir(DATADIR):
      try :
         img_array = add_noise("s&p", cv2.imread(os.path.join(DATADIR, img), cv2.IMREAD_GRAYSCALE))
         new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
         training_data.append([new_array, class_num])
      except Exception as e:
         pass

create_training_data()

random.shuffle(training_data)

X = [] #features

for features, _ in training_data:
	X.append(features)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE)

# Creating the files containing all the information about your model
pickle_out = open("noise.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()