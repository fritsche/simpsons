# Data Augmentation
# http://docs.opencv.org/master/da/d6e/tutorial_py_geometric_transformations.html#gsc.tab=0

import cv2
import os
import sys

for filename in os.listdir('data/Train/'):

	path = 'data/Train/'+filename
	img = cv2.imread(path)
	img = cv2.resize(img, (150,150), interpolation = cv2.INTER_LINEAR)
	rows, cols, channels = img.shape

	M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
	dst = cv2.warpAffine(img,M,(cols,rows))
	cv2.imwrite('data/TrainExt/'+filename+'90.bmp',dst)

	M = cv2.getRotationMatrix2D((cols/2,rows/2),180,1)
	dst = cv2.warpAffine(img,M,(cols,rows))
	cv2.imwrite('data/TrainExt/'+filename+'180.bmp',dst)

	M = cv2.getRotationMatrix2D((cols/2,rows/2),270,1)
	dst = cv2.warpAffine(img,M,(cols,rows))
	cv2.imwrite('data/TrainExt/'+filename+'270.bmp',dst)	

	M = cv2.getRotationMatrix2D((cols/2,rows/2),0,1)
	dst = cv2.warpAffine(img,M,(cols,rows))
	cv2.imwrite('data/TrainExt/'+filename+'0.bmp',dst)	
