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



