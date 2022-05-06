import cv2
import numpy as np
import sys
import argparse

# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", type=str, required=True,
# 	help="path to input image")
# args = vars(ap.parse_args())


# load the image, convert it to grayscale, and blur it slightly
# image = cv2.imread(args["image"])
image = cv2.imread('1.jpg')
grey = image[:, :, 2]		#sets to the 3rd channel of input (greyscale)
thresh = cv2.threshold(grey, 0, 55, cv2.THRESH_BINARY)[1]		#converts greyscale to binary
blurred = cv2.GaussianBlur(thresh, (5, 5), 0)
# show the original and blurred images
cv2.imshow("Original", image)
cv2.imshow("Grey", grey)
cv2.imshow("Thresh", thresh)
cv2.imshow("Blurred", blurred)
cv2.waitKey()




# grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# blurred = cv2.GaussianBlur(grey, (5, 5), 0)
# frame = cv2.imread('1.jpg')
# cv2.namedWindow("Raw", cv2.WINDOW_NORMAL)
# cv2.namedWindow("blurred", cv2.WINDOW_NORMAL)
# cv2.imshow("Raw", frame)
# cv2.imshow("blurred", blurred)

