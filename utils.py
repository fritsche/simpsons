# Color histogram
#
# Created by gmfritsche based on
# http://www.pyimagesearch.com/2014/03/03/charizard-explains-describe-quantify-image-using-feature-vectors/

import cv2

def load_image (path):
	image = cv2.imread(path)
	return image
