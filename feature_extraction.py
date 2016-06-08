# Color histogram
#
# Created by gmfritsche based on
# http://www.pyimagesearch.com/2014/03/03/charizard-explains-describe-quantify-image-using-feature-vectors/

import cv2
import sys
import os
import numpy as np
from skimage.feature import hog

def color_histogram (image):
	hist = cv2.calcHist([image], [0, 1, 2], None, [4, 4, 4], [0, 256, 0, 256, 0, 256])
	hist = hist.flatten()
	return hist

def hu_moments (image):
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	features = cv2.HuMoments(cv2.moments(image)).flatten()
	return features

def histogram_of_oriented_gradients (image):
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image = cv2.resize(image, (150,150), interpolation = cv2.INTER_AREA)
	hist = hog(image, orientations=8, pixels_per_cell=(16,16),
				cells_per_block=(1, 1), visualise=False, normalise=False)
	return hist.flatten()

def extraction (pathtodata, descriptor) :

	features = []
	classes = []
	
	for filename in os.listdir(pathtodata):
		if filename.startswith ("bart"):
			classes.append(0)
		elif filename.startswith ("homer"):
			classes.append(1)
		elif filename.startswith ("lisa"):
			classes.append(2)
		elif filename.startswith ("maggie"):
			classes.append(3)
		elif filename.startswith ("marge"):
			classes.append(4)
		else:
			classes.append(5)

		image = cv2.imread(pathtodata+"/"+filename)

		image = cv2.resize(image, (150,150), interpolation = cv2.INTER_LINEAR)

		vector = []

		if (descriptor == "color_histogram"):
			vector = color_histogram (image)
		elif (descriptor == "hu_moments"):
			vector = hu_moments (image)
		elif (descriptor == "histogram_of_oriented_gradients"):
			vector = histogram_of_oriented_gradients (image)
		else :
			print "Unknown descriptor", descriptor
		
		features.append(vector)

	return features, classes

if __name__=='__main__':

	features = []
	classes = []

	if __name__ == "__main__":

		pathtodata = sys.argv[1]
		descriptor = sys.argv[2]

		features, classes = extraction (pathtodata, descriptor)

		np.savetxt("data/features.txt", features)
		np.savetxt("data/classes.txt", classes, fmt="%d")
