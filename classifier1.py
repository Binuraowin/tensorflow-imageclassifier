import ssl
import tensorflow as tf
import numpy as np
from tensorflow import keras
##matplotlip for display  image
import matplotlib.pyplot as plt

## pre-defeined dataset
fashion_mnist = keras.datasets.fashion_mnist

(train_images,train_labels),(test_images,test_labels) = fashion_mnist.load_data()

print(test_labels[0])
plt.imshow(test_images[8],cmap='gray',vmin=0,vmax=255)
plt.show()
