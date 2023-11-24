import os
import numpy as np
import keras.utils as image
from keras.utils.np_utils import to_categorical
from recordings_to_spectograms import trasnform_train_data
import tensorflow as tf



#  make the spectrograms of the data from the DATASET
trasnform_train_data()

#  split the samples to test and train sets
imagesDir = 'C:\\Users\\spirt\\Desktop\\pikrakis\\spectrograms\\'
trainset = []
testset = []
for file in os.listdir(imagesDir):
  label = file.split('_')[0]
  sample_number = file.split('_')[2]
  img = tf.keras.utils.load_img(imagesDir+file)
  if sample_number in ['0.png','1.png','2.png','3.png','4.png','5.png','6.png','7.png','8.png','9.png']:
    testset.append([image.img_to_array(img), label])
  else:
    trainset.append([image.img_to_array(img), label])


# Get only images in the train list not the Labels
X_train = [item[0] for item in trainset]
# Get only Labels in the train list not the images
y_train = [item[1] for item in trainset]
# Get only images in the test list not the Labels
X_test = [item[0] for item in testset]
# Get only Labels in the test list not the images
y_test = [item[1] for item in testset]

print(X_train)
print(y_train)

print(X_test)
print(y_test)
# Convert list to numpy array in order to define input shape
X_train = np.asanyarray(X_train)
y_train = np.asanyarray(y_train)
X_test = np.asanyarray(X_test)
y_test = np.asanyarray(y_test)


# convert to one hot representation
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# Save data to file to load without creating datasets again
np.save('xtrain_file', X_train)
np.save('xtest_file', X_test)
np.save('ytrain_file', y_train)
np.save('ytest_file', y_test)
