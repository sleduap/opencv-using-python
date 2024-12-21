import cv2
import numpy as np
import matplotlib.pyplot as plt 

plt.rcParams['image.cmap'] = 'gray'


# Read image in grayscale.
img = cv2.imread('road_lanes.png', cv2.IMREAD_GRAYSCALE)

# Perform binary thresholding.
retval, img_thresh = cv2.threshold(img, 165, 255, cv2.THRESH_BINARY)

# Display the images.
plt.figure(figsize = [20, 8])
plt.subplot(121); plt.imshow(img); plt.title('Original')
plt.subplot(122); plt.imshow(img_thresh); plt.title('Thresholded')