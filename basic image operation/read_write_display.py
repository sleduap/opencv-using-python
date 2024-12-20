import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# Tells notebook to render figures in-page.
  
from IPython.display import Image

# Display 18x18 pixel image.
Image(filename='img_bw_18x18.png')
"""retval = cv2.imread(filename[, flags])
retval: Is the image if it is successfully loaded. Otherwise, it is None. This may happen if the filename is wrong or the file is corrupt.

The function has 1 required input argument and one optional flag:

filename: This can be an absolute or relative path. This is a mandatory argument.
flags: These flags are used to read an image in a particular format (for example, grayscale/color/with alpha channel). This is an optional argument with a default value of cv2.IMREAD_COLOR or 1 which loads the image as a color image.
Before we proceed with some examples, let's also have a look at some of flags available.

cv2.IMREAD_GRAYSCALE or 0: Loads image in grayscale mode
cv2.IMREAD_COLOR or 1: Loads a color image. Any transparency of image will be neglected. It is the default flag.
cv2.IMREAD_UNCHANGED or -1: Loads image using its original channels, which could include the alpha channel.
    """

# Read image as grayscale.
bw_img = cv2.imread('img_bw_18x18.png', cv2.IMREAD_GRAYSCALE)

# Print the image data (pixel values) of a 2D NumPy array.
# Each pixel value is 8-bits in the range [0, 255].
print(bw_img)

# Print the size of image.
print('Image size is ', bw_img.shape)

# Print data-type of image.
print('Data type of image is ', bw_img.dtype)


# Display the image.
plt.imshow(bw_img);

# Set color map to gray scale for proper rendering.
plt.imshow(bw_img, cmap = 'gray');
# matplotlib.rc('image', cmap = 'gray')

# Read image as gray scale.
MNIST_3_img = cv2.imread('MNIST_3_18x18.png', cv2.IMREAD_GRAYSCALE)

# Display the image.
plt.imshow(MNIST_3_img, cmap = 'gray');

#Print the image matrix data
print(MNIST_3_img)


# read and write
image = cv2.imread('Apollo-8-Launch.jpg')
plt.figure(figsize = [10, 10])
plt.imshow(image)
plt.title('Apollo-8-Launch.jpg', fontsize = 16);


cv2.imwrite('Apollo-8-Launch.png', image)