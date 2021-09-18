# https://towardsdatascience.com/all-the-steps-to-build-your-first-image-classifier-with-code-cf244b015799

import numpy as np
import os
from matplotlib import pyplot as plt
import cv2
import random
import pickle

from variables import FOLDERS, DIMENSIONS
IMG_SIZE = DIMENSIONS

print("Running: data_image_process.py at "+str(IMG_SIZE)+"x"+str(IMG_SIZE))

file_list = []
class_list = []

DATADIR = "data"

# All the categories you want your neural network to detect
CATEGORIES = FOLDERS # "plzebraPZGZ" #["absent","noise","present"] 

# Checking or all images in the data folder
for category in CATEGORIES :
	path = os.path.join(DATADIR, category)
	for img in os.listdir(path):
		img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)

training_data = []

def create_training_data():
	for category in CATEGORIES :
		path = os.path.join(DATADIR, category)
		class_num = CATEGORIES.index(category)
		for img in os.listdir(path):
			try :
				img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
				new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
				training_data.append([new_array, class_num])
			except Exception as e: pass

create_training_data()

random.shuffle(training_data)

X = [] #features
y = [] #labels

for features, label in training_data:
	X.append(features)
	y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE)

# Creating the files containing all the information about your model
pickle_out = open("images.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("labels.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()
