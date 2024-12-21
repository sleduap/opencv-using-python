import cv2
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib


yellow  = (0, 255, 255)
red     = (0, 0, 255)
magenta = (255, 0, 255)
green   = (0, 255, 0)


# Read the image.
image = cv2.imread('Apollo-8-Launch.jpg', cv2.IMREAD_COLOR)

# Display the original image.
plt.figure(figsize = [7, 7])
plt.imshow(image);
print(image.shape)


# Make a copy of the original image.
image_line = image.copy()

# Draw a yellow line (using: cv2.LINE_8)
image_line = cv2.line(image_line, (0, 445), (450, 465), yellow, thickness=3, lineType=cv2.LINE_8)

# Display the annotated image.
plt.figure(figsize = [10, 10])
plt.imshow(image_line[:, :, ::-1]);


# Make a copy of the original image.
image_circle = image.copy()

# Draw a red circle.
image_circle = cv2.circle(image_circle, (195, 55), 20, red, thickness = 2, lineType = cv2.LINE_AA)

# Display the annotated image.
plt.figure(figsize = [8, 8])
plt.imshow(image_circle[:, :, ::-1]);


# Make a copy of the original image.
image_rectangle = image.copy()

# Draw a magenta rectangle.
image_rectangle = cv2.rectangle(image_rectangle, (300, 150), (480, 420), magenta, thickness = 3, lineType = cv2.LINE_8)

# Display the annotated image.
plt.figure(figsize = [8, 8])
plt.imshow(image_rectangle[:, :, ::-1]);