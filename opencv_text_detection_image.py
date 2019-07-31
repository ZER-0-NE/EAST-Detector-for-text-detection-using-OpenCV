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

# grab the rows and columns from score volume
(numrows, numcols) = scores.shape[2:4]
rects = [] #stores the bounding box coordiantes for text regions
confidences = [] # stores the probability associated with each bounding box region in rects

for y in range(0, numrows):
	scoresdata = scores[0,0,y]
	xdata0 = geometry[0,0,y]
	xdata1 = geometry[0,1,y]
	xdata2 = geometry[0,2,y]
	xdata3 = geometry[0,3,y]
	anglesdata = geometry[0,4,y]

	for x in range(0, numcols):
		if scoresdata[x] <args["min_confidence"]: # if score is less than min_confidence, ignore
			continue
	
	(offsetx, offsety) = (x*4.0, y*4.0) # EAST detector automatically reduces volume size as it passes through the network
	#extracting the rotation angle for the prediction and computing their sine and cos

	angle = anglesdata[2:4]
	cos = np.cos(angle)
	sin = np.cos(angle)

	h = xdata0[x] + xdata2[x]
	w = xdata1[x] + xdata3[x]

	endx = int(offsetx + (cos * xdata1[x]) + (sin * xdata2[x]))
	endy = int(offsety + (sin * xdata1[x]) + (cos * xdata2[x]))
	startx = int(endx - w)
	starty = int(endy - h)

	startx, endx = np.clip([startx, endx], 0, w)
	starty, endy = np.clip([starty, endy], 0, h)

	# appending the confidence score and probabilities to list
	rects.append((startx, starty, endx, endy))
	confidences.append(scoresdata[x])

# applying non-maxima suppression to supppress weak and overlapping bounding boxes
boxes = non_max_suppression(np.array(rects), probs = confidences)

for(startx, starty, endx, endy) in boxes:
	startx = int(startx * rW)
	starty = int(starty * rH)
	endx = int(endx * rW)
	endy = int(endy * rH)

	cv2.rectangle(orig, (startx, starty), (endx, endy), (0,255,0), 2)

cv2.imshow("text Detection", orig)
cv2.waitKey(0)
