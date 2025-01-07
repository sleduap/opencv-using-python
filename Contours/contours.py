import cv2
import matplotlib.pyplot as plt
import numpy as np
# %matplotlib inline

import matplotlib
matplotlib.rcParams['figure.figsize'] = [20.0, 10.0]
matplotlib.rcParams['image.cmap'] = 'gray'


imagePath = 'shapes.jpg'
image = cv2.imread(imagePath)
# Convert to grayscale
imageGray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

# Display image
plt.imshow(imageGray);


ret, thresh = cv2.threshold(imageGray, 200, 255, cv2.THRESH_BINARY_INV)
plt.imshow(thresh);


# Find all contours in the image.
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# Number of contours.
print("Number of contours found = {}".format(len(contours)))

# Hierarchy.
print("\nHierarchy : \n{}".format(hierarchy))



# Create a copy of the original image.
imageCopy1 = image.copy()
# Draw all the contours.
cv2.drawContours(imageCopy1, contours, -1, (0,0,255), 3)
plt.imshow(imageCopy1[:,:,::-1]);

# Find external contours in the image.
# contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
print("Number of contours found = {}".format(len(contours)))

# Create a copy of the original image.
imageCopy2 = image.copy()

# Draw all the contours.
cv2.drawContours(imageCopy2, contours, -1, (0,0,255), 4)

# Display.
plt.imshow(imageCopy2[:,:,::-1]);

# Create a copy of the original image.
imageCopy3 = image.copy()
# Draw contours.
cv2.drawContours(imageCopy3, contours[3], -1, (0,0,255), 4)
# Display.
plt.imshow(imageCopy3[:,:,::-1]);


def convert_color(hsv):
    """Utility to convert a single hsv color tuple into bgr"""
    pixel_img = np.uint8([[hsv]])
    print("a",pixel_img)
    return tuple(int(i) for i in cv2.cvtColor(pixel_img, cv2.COLOR_HSV2BGR).flatten())

imageCopy4 = image.copy()

for i, single_contour in enumerate(contours):
    hsv = (int(i/len(contours) * 180), 255, 255)
    color = convert_color(hsv)
    cv2.drawContours(imageCopy4, contours, i, color, 3)
    plt.imshow(imageCopy4[:,:,::-1]);

# Find all contours in the image.
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# Create a copy of the original image.
imageCopy5 = image.copy()

# Draw all the contours.
cv2.drawContours(imageCopy5, contours, -1, (0,0,255), 3)


for cnt in contours:
    # We will use the contour moments
    # to find the centroid.
    M = cv2.moments(cnt)
    x = int(round(M["m10"]/M["m00"]))
    y = int(round(M["m01"]/M["m00"]))
    
    # Mark the center.
    cv2.circle(imageCopy5, (x,y), 10, (255,0,0), -1)
    
# Display.
plt.imshow(imageCopy5[:,:,::-1]);


for index,cnt in enumerate(contours):
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    print("Contour #{} has area = {} and perimeter = {}".format(index+1,area,perimeter))
    
    
# Create a copy of the original image.
imageCopy6 = image.copy()
for cnt in contours:
    # Vertical rectangle.
    x,y,w,h = cv2.boundingRect(cnt)
    cv2.rectangle(imageCopy6, (x,y), (x+w,y+h), (0,255,0), 4)
# Display.
plt.imshow(imageCopy6[:,:,::-1]);

# Create a copy of the original image.
imageCopy7 = image.copy()
for cnt in contours:
    # Rotated bounding box
    box = cv2.minAreaRect(cnt)
    boxPts = np.intp(cv2.boxPoints(box))
    # Draw contours.
    cv2.drawContours(imageCopy7, [boxPts], -1, (0,255,0), 4)
plt.imshow(imageCopy7[:,:,::-1]);


# Create a copy of the original image.
imageCopy8 = image.copy()
for cnt in contours:
    # Circle fitting.
    (x,y),radius = cv2.minEnclosingCircle(cnt)
    cv2.circle(imageCopy8, [int(x),int(y)], int(round(radius)), (0,255,0), 4)
plt.imshow(imageCopy8[:,:,::-1]);



# Create a copy of the original image.
imageCopy9 = image.copy()
for cnt in contours:
    if len(cnt) < 5:
        continue
    # Ellipse fitting.
    ellipse = cv2.fitEllipse(cnt)
    cv2.ellipse(imageCopy9, ellipse, (0,255,0), 4)
plt.imshow(imageCopy9[:,:,::-1]);