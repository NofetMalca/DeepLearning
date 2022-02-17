
import numpy as np
import matplotlib.pyplot as plt
import cv2
import h5py
from keras.models import Sequential, model_from_json,Model
from keras.layers import *
from keras.utils import np_utils
from tensorflow.keras.optimizers import Adam , SGD 
from collections import Counter
from scipy import ndimage
import csv


test_path = 'SynthText_test.h5'

json_model_path = 'final_model.json'
h5_weights_path = 'final_model.h5'
csv_path = 'predictionsFonts.csv'

#test_path = '/content/drive/MyDrive/testProject/SynthText_test.h5'

#json_model_path = '/content/drive/MyDrive/ProjectComputerVision/final_model.json'
#h5_weights_path = '/content/drive/MyDrive/ProjectComputerVision/final_model.h5'
#csv_path = '/content/drive/MyDrive/testProject/predictionsFonts.csv'

#Names of all fonts
class_names = {'Raleway': 0,  'Open Sans': 1, 'Roboto': 2,'Ubuntu Mono': 3,'Michroma': 4, 'Alex Brush': 5, 'Russo One': 6 }

HT = 40 # height 40 of each image
WD = 30 # width 30 of each image

X_test = [] # images test
y_test = [] # labels test
chars = []  # for each letter for .csv
images_names = [] # for each מame of image for .csv

#file of database
ts = h5py.File(test_path, 'r')
tst_names = list(ts['data'].keys())

#read test data
for im in tst_names:
    img = ts['data'][im][:]
    txt = ts['data'][im].attrs['txt']
    charBB = ts['data'][im].attrs['charBB']
    wordBB = ts['data'][im].attrs['wordBB']
    txtAsString = ''.join([w.decode('UTF-8') for w in txt])

    for i, bb in enumerate(charBB.transpose(2, 0, 1)):
        srcPts = np.float32([bb.T[0], bb.T[1], bb.T[3], bb.T[2]])
        dstPts = np.float32([[0, 0], [WD, 0], [0, HT], [WD, HT]])
        warpMatrix = cv2.getPerspectiveTransform(srcPts, dstPts)
        letterImg = cv2.warpPerspective(img, warpMatrix, (WD, HT))
        X_test.append(cv2.cvtColor(letterImg, cv2.COLOR_BGR2GRAY))
        chars.append(txtAsString[i])
        images_names.append(im)

X_test = np.array(X_test)

# add color channel
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1],
                                  X_test.shape[2], 1)
#print(X_test.shape)

# make data between 0 and 1
X_test = X_test.astype('float32')
X_test /= 255

#test minus mean of train data(after calculated the average of all the pictures-0.461)
mean = np.ndarray((HT,WD,1))
mean.fill(0.461)
X_test = X_test - mean

# read model and weightd from json and h5 files
json_file = open(json_model_path, 'r')
loaded_model_json = json_file.read()
json_file.close()
finalModel = model_from_json(loaded_model_json)
finalModel.load_weights(h5_weights_path)

# predict test
y_test = np.argmax(finalModel.predict([X_test,X_test]), axis=-1)

#write predictions to csv file
with open(csv_path, mode='w', newline='') as csv_file:
          writer = csv.writer(csv_file)
          writer.writerow(['','image','char',"b'Raleway'",
                                      "b'Open Sans'","b'Roboto'","b'Ubuntu Mono'","b'Michroma'","b'Alex Brush'","b'Russo One'"])
          for i in range(len(y_test)):
              v = [0, 0, 0, 0, 0, 0, 0]
              v[y_test[i]] = 1           
              row = [i, images_names.pop(0), chars[i], v[0], v[1], v[2], v[3], v[4], v[5], v[6]]
              writer.writerow(row)