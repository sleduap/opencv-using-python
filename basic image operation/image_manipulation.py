import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from IPython.display import Image
plt.rcParams['image.cmap'] = 'gray'


# Read image as grayscale.
mnist_3_img = cv2.imread('MNIST_3_18x18.png', cv2.IMREAD_GRAYSCALE)

# Display the image.
plt.figure(figsize = (4, 4))
plt.imshow(mnist_3_img);

print(mnist_3_img[3, 10])
print(mnist_3_img[12, 2])
print(mnist_3_img[10, 12])



# 1.2 Modifying Image Pixels
# Make a copy of the original image.
mnist_3_img_copy = mnist_3_img.copy()

# Modify four pixels.
mnist_3_img_copy[2, 2] = 100
mnist_3_img_copy[2, 3] = 125
mnist_3_img_copy[3, 2] = 150
mnist_3_img_copy[3, 3] = 175

# Use numPy array slicing to modify a group of pixels.
mnist_3_img_copy[0:17,16] = 150

# Print the image data.
print(mnist_3_img_copy)

# Display the modified image.
plt.figure(figsize = (4, 4))
plt.imshow(mnist_3_img_copy);


# 2. Cropping Images
img_eagle = cv2.imread('Eagle_in_Flight.jpg', cv2.IMREAD_COLOR)

plt.figure(figsize = (8, 8))
plt.imshow(img_eagle[:, :, ::-1]);


cropped_region = img_eagle[20:420, 150:600]
plt.imshow(cropped_region[:, :, ::-1]);


# 3. Resizing Images

#3.1 Method 1: Specifying Scaling Factor using fx and fy
# Resize the image.
resized_cropped_region_2x = cv2.resize(cropped_region, None, fx = 2, fy = 2)

# Display the resized image.
plt.imshow(resized_cropped_region_2x[:, :, ::-1]);

# 






#3.2 Method 2: Specifying Exact Size of the Output Image
desired_width  = 400
desired_height = 200
dim = (desired_width, desired_height)

resize_exact = cv2.resize(cropped_region, dim, interpolation = cv2.INTER_AREA)

#3.3 Method 3: Resize Dimension while Maintaining Aspect Ratio
# Method 2: Using 'dsize'.
desired_width = 200
aspect_ratio = cropped_region.shape[1] / cropped_region.shape[0]
desired_height = int(desired_width * aspect_ratio)
dim = (desired_width, desired_height)

# Resize the image.
resized_cropped_region = cv2.resize(cropped_region, dsize = dim, interpolation = cv2.INTER_AREA)
plt.figure(figsize = (6, 6))
plt.imshow(resized_cropped_region[:, :, ::-1]);


# Flip the image three ways.
img_eagle_flipped_horz = cv2.flip(img_eagle, 1)
img_eagle_flipped_vert = cv2.flip(img_eagle, 0)
img_eagle_flipped_both = cv2.flip(img_eagle, -1)

# Dispay the images.
plt.figure(figsize = [18, 5])
plt.subplot(141); plt.imshow(img_eagle_flipped_horz[:, :, ::-1])
plt.title('Horizontal Flip')
plt.subplot(142); plt.imshow(img_eagle_flipped_vert[:, :, ::-1])
plt.title('Vertical Flip')
plt.subplot(143); plt.imshow(img_eagle_flipped_both[:, :, ::-1])
plt.title('Both Flipped')
plt.subplot(144); plt.imshow(img_eagle[:, :, ::-1])
plt.title('Original');