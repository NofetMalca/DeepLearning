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
from sklearn.model_selection import train_test_split


#Names of all fonts
class_names = {'Raleway': 0,  'Open Sans': 1, 'Roboto': 2,'Ubuntu Mono': 3,'Michroma': 4, 'Alex Brush': 5, 'Russo One': 6 }
   
X_train =[] # images train
y_train =[] # labels train
X_valid =[] # images validation
y_valid =[] # labels validation
X_test =[]  # images test
y_test =[]  # labels test


HT = 40  # height 40 of each image
WD = 30  # width 30 of each image

#file of database
db1 = h5py.File('/content/drive/MyDrive/ProjectComputerVision/SynthText.h5', 'r')
data1_names = list(db1['data'].keys())

# Build train set
for db, data_names in zip([db1], [data1_names]):
    for im in data_names:
        img = db['data'][im][:]
        font = db['data'][im].attrs['font']
        txt = db['data'][im].attrs['txt']
        charBB = db['data'][im].attrs['charBB']
        # wordBB = db['data'][im].attrs['wordBB']

        for i, bb in enumerate(charBB.transpose(2, 0, 1)):
            srcPts = np.float32([bb.T[0], bb.T[1], bb.T[3], bb.T[2]])# ndarray.T The transposed array. [[1,2,3]] -> [[1][2][3]]
            dstPts = np.float32([[0, 0], [WD, 0], [0, HT], [WD, HT]])
            #getPerspectiveTransform-associated with the change in the viewpoint
            warpMatrix = cv2.getPerspectiveTransform(srcPts, dstPts)
            #warpPerspective-returns an image or video whose size is the same as the size of the original image or vide
            letterImg = cv2.warpPerspective(img, warpMatrix, (WD, HT))
            gray = cv2.cvtColor(letterImg, cv2.COLOR_BGR2GRAY) #change to grayScale
            r = np.random.randint(-2, 1)
            '''
			For more data images-
            The array is rotated in the plane defined by the two axes given by 
            the axes parameter using spline interpolation of the requested order.
            '''
            X_train.append(ndimage.rotate(gray, r, mode='reflect', reshape=False))
            X_train.append(ndimage.rotate(gray, r + 2, mode='reflect', reshape=False))
            y_train.append(class_names[font[i].decode('UTF-8')])
            y_train.append(class_names[font[i].decode('UTF-8')])

X_train = np.array(X_train)
y_train = np.array(y_train)

# add color channel
X_train = X_train.reshape(X_train.shape[0],
                                    X_train.shape[1],
                                    X_train.shape[2], 1)

# make data between 0 and 1
X_train = X_train.astype('float32')
X_train /= 255

# calculate mean image of the train data,did not use because i saw that global mean is 0.461
# mean = np.zeros((HT, WD, 1))
# for i, row in enumerate(X_train.transpose(1, 2, 3, 0)):
#     for j, pxl in enumerate(row):
#         for k, colorInPxl in enumerate(pxl):
#             mean[i][j][k] = colorInPxl.mean()

# minus mean from train date
mean = np.ndarray((HT,WD,1))
mean.fill(0.461)
X_train = X_train - mean

# Binarization of labels
y_train = np_utils.to_categorical(y_train, 7)#spilit to 7 category
#print(y_train[:6])

X_train, X_test, y_train, y_test = train_test_split(X_train,y_train, train_size = 0.8)
X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, train_size =0.5 )


# my first model
model1 = Sequential()
model1.add(Convolution2D(128, 3, 3, activation='relu', input_shape=(HT,WD,1)))
model1.add(Convolution2D(128, 3, 3, activation='relu'))
model1.add(Dropout(0.5))#for half of the layers - does not relate
model1.add(BatchNormalization())#Average 0

model1.add(Flatten())#flattens the multi-dimensional input tensors into a single dimension
model1.add(Dense(256, activation='relu'))# Layer activation functions, Model each model from a router to 128 and a router for real answers.
model1.add(Dropout(0.5))
model1.add(Dense(7, activation='softmax'))#7 category
model1.add(BatchNormalization())

# second model(after my changes and optimizations by github)
model2=Sequential()

model2.add(Convolution2D(32, kernel_size=(7, 7), activation='relu', input_shape=(HT,WD,1)))
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(BatchNormalization())

model2.add(Convolution2D(64, kernel_size=(5, 5), activation='relu'))
model2.add(BatchNormalization())

model2.add(Conv2DTranspose(32, (5,5), strides = (2,2), activation = 'relu', padding='same', kernel_initializer='uniform'))
model2.add(UpSampling2D(size=(2, 2)))
model2.add(BatchNormalization())

model2.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model2.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model2.add(Dropout(0.5))
model2.add(BatchNormalization())

model2.add(Flatten())
model2.add(Dense(256, activation='relu'))
model2.add(Dropout(0.5))
model2.add(BatchNormalization())
model2.add(Dense(128,activation='relu'))
model2.add(Dropout(0.5))
model2.add(Dense(7, activation='softmax'))#7 category
model2.add(BatchNormalization())

# concatenate the 2 models
modeltmp = concatenate([model1.output, model2.output], axis=-1)# input a list of tensors (same shape) returns a single tensor that is the concatenation of all inputs.
modeltmp = Dense(7, activation='sigmoid')(modeltmp)##7 category
finalModel = Model(inputs=[model1.input, model2.input], outputs=modeltmp)
opt = Adam(lr=0.0001)#adaptive learning rate optimization algorithm
finalModel.compile(optimizer=opt, loss='mean_squared_error', metrics=['accuracy'])

# fitting
history = finalModel.fit([X_train,X_train], y_train, batch_size=64, epochs=60,verbose=1, validation_data=([X_valid,X_valid], y_valid))

#graph Model
plt.plot(history.history['loss'], label='MSE (training data)')
plt.plot(history.history['val_loss'], label='MSE (validation data)')
plt.title('MSE for Chennai Reservoir Levels')
plt.ylabel('MSE value')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
plt.show()

# save model to some path
model_path = "/content/drive/MyDrive/ProjectComputerVision/final_model.json"
weights_path = "/content/drive/MyDrive/ProjectComputerVision/final_model.h5"
model_json = finalModel.to_json()
with open(model_path, "w") as json_file:
    json_file.write(model_json)
finalModel.save_weights(weights_path)

#SCORE
print(finalModel.evaluate([X_test,X_test],y_test,verbose=0))
print(finalModel.metrics_names)

#visualModel 
from keras.utils.vis_utils import plot_model
plot_model(finalModel, to_file='/content/drive/MyDrive/ProjectComputerVision/model_visual.png', show_shapes=True, show_layer_names=True)