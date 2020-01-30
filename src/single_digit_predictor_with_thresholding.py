import cv2
import numpy as np

import time
from keras.models import load_model
vidcap = cv2.VideoCapture(0)
count=0
success,fimage = vidcap.read()
img_size = (224,224)
for c in range(0,20):
	success,fimage = vidcap.read()
cv2.imwrite("fram%d.jpg" % count, fimage)
fimage = cv2.cvtColor(fimage, cv2.COLOR_BGR2GRAY)
fimage = fimage[100:350, 100:350]

while (vidcap.isOpened()):
	success,cimage = vidcap.read()
	
	cv2.rectangle(cimage, (350,350), (100,100), (0,255,0),0)
	cv2.imshow('Gesture', cimage)
	cv2.waitKey(50)
	mimage = cimage[100:350, 100:350]
	print(mimage.shape)
	image = cv2.cvtColor(mimage, cv2.COLOR_BGR2GRAY)
	diff = cv2.absdiff(fimage,image)
	diff = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
	#cv2.imshow('3',diff)
	op = cv2.sumElems(diff)
	print (op[0])
	cv2.imshow('Gesture', cimage)
	cv2.waitKey(50)


	if (op[0]>2000000):
		for c in range(0,46):
			success,fimage = vidcap.read()
			cv2.rectangle(cimage, (350,350), (100,100), (0,255,0),0)
			cv2.imshow('Gesture', cimage)
		#time.sleep(5)
		cv2.imshow('Gesture', cimage)		
		success,cimage = vidcap.read()
		cv2.imshow('Gesture', cimage)
		count=1
		mimage = cimage[100:350, 100:350]
		print(mimage.shape)
		#mimage = cv2.cvtColor(mimage, cv2.COLOR_BGR2GRAY)
		cv2.imwrite("fram%d.png" % count, mimage)     # save frame as JPEG file
		break
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
