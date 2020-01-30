import cv2
import numpy as np
import keras
import keras.backend as TF
import time
from keras.models import load_model
import tensorflow as tf


vidcap = cv2.VideoCapture(0)
count=0
success,fimage = vidcap.read()
img_size = (224,224)
font = cv2.FONT_HERSHEY_SIMPLEX

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config = config)

for c in range(0,20):
	success,fimage = vidcap.read()
cv2.imwrite("fram%d.jpg" % count, fimage)
fimage = cv2.cvtColor(fimage, cv2.COLOR_BGR2GRAY)


for c in range(60,0,-1):
	success,fimage = vidcap.read()
	cv2.waitKey(48)
	#cv2.imshow('Gesture', fimage)
	
	print(c)
	cv2.rectangle(fimage, (350,350), (100,100), (0,255,0),0)
	cv2.putText(fimage, 'Do the Gesture', (230, 50), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
	cv2.putText(fimage, str(c), (50, 50), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
	cv2.imshow('Gesture', fimage)
print('done')
mimage = fimage[100:350, 100:350]

model = load_model('drivev1_2(224,224_10).hdf5')
model.summary()
mimage = cv2.resize(mimage,img_size)
img_list=[]
img_list.append(mimage)
img_list.append(mimage)
img_data = np.array(img_list)
img_data = img_data.astype('float32')
#print(img_data.shape)
img_data = img_data.reshape(img_data.shape)# + (1,))
print(img_data.shape)
k=model.predict(img_data,verbose=1)
print(k)
print((np.argmax(k[0])))
