import cv2 as cv 
import numpy as np 
import matplotlib.pyplot as plt 

# Input the path to the image
path_2_img = input("Enter the proper path to the image : ")


# Reading the given image
img = cv.imread(path_2_img)

# Printing the image
cv.imshow("Given Image ", img)

# Preprocessing 
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
blur = cv.medianBlur(gray, 17)
# Thresholding to identify the region of the character
ret, thresh = cv.threshold(blur, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

# creating the kernel 
kernel = np.ones((3,3),np.uint8)

# Finding an opening for the character and applying morphological operations
opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)

# dialting to find the area which we are sure that that is the background
sure_bg = cv.dilate(opening, kernel, iterations=3)


# Eroding the opening the determine the sure foreground
eroded = cv.erode(opening,kernel)
ret, sure_fg = cv.threshold(eroded, 0.7*eroded.max(),255,0)
sure_fg = np.uint8(sure_fg)

# finding the unknown region where wwe don't know what is there
unknown = cv.subtract(sure_bg,sure_fg)

# Marking the elements in sure foreground and setting the background as 0
ret,markers = cv.connectedComponents(sure_fg)

# adding 1 to the marker so that background is not 0
markers = markers + 1
markers[unknown == 255] = 0

# Using the watershed algorithm to mark the characters
markers = cv.watershed(img,markers)

# coloring the unknown area as blue
img[markers == -1] = [255,0,0]

# coloring the background as black
img[markers == 1] = [0,0,0]

# Finally showing the segmented character
cv.imshow("Segmented Character", img)




cv.waitKey(0)
cv.destroyAllWindows()