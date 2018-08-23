import numpy as np
import cv2
import argparse
import time
from imutils.object_detection import non_max_suppression


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type = str,
	 help = "path to input image")
ap.add_argument("-east", "--east", type = str,
	 help = "path to input EAST Detector")
ap.add_argument("-c", "--min-confidence", type = float,
	default = 0.5, help = "minimum probability required to inspect a region")
ap.add_argument("-w", "--width", type = int,
	default = 320, help = "resized image width(should be multiple of 32)")
ap.add_argument("-e", "--height", type = int,
	default = 320, help = "resized image height(should be multiple of 32)")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
orig = image.copy()
(h,w) = image.shape[:2]

# setting new width and height
(newW, newH) = (args["width"], args["height"])
rW = w/float(newW)
rH = h/float(newH)

#resize the image
image = cv2.resize(image, (newW, newH))
(h,w) = image.shape[:2]

# we define the two output layer names for the EAST Detector model
# that we are interested in
layers = [
	"feature_fusion/Conv_7/Sigmoid",
	"feature_fusion/concat_3"]

# loading pre-trained EAST detector
print("[INFO] loading EAST text detector")
net = cv2.dnn.readNet(args["east"])

blob = cv2.dnn.blobFromImage(image, 1.0, (w,h),
	(123.68, 116.78, 103.94), swapRB = True, crop = False)

start = time.time()
net.setInput(blob)
(scores, geometry) = net.forward(layers)
end = time.time()

print("[INFO] text detection took {:.6f} seconds".format(end-start))


















