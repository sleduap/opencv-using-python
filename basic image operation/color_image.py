import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from IPython.display import Image
plt.rcParams['image.cmap'] = 'gray'


# Read the image.
logo = 'facebook_logo.jpg'
logo_img = cv2.imread(logo, cv2.IMREAD_COLOR)

# Print the size of the image.
print("Image size is ", logo_img.shape)

plt.figure(figsize = (10, 10))
plt.imshow(logo_img);


"""What happened to the color?¶
The color displayed above is different from the actual image. 
This is because matplotlib expects the image to be in RGB format whereas OpenCV stores images in BGR format. 
Thus, for correct display, we need to reverse the channel order of the image in order to properly render 
the color of the image.
""" """
Swap the Red and Blue channels
There are a couple of different approaches to reversing the order of the color channels. 
The first approach shown below uses a short-hand NumPy array slicing syntax that will reverse the order of 
the channels in the 3rd dimension of the image array."""

# Swap the Red and Blue channels.
# Swap the Red and Blue color channels.
logo_img = logo_img[:, :, ::-1]

# Display the image.
plt.figure(figsize = (10, 10))
plt.imshow(logo_img);

########################################################################################
# Read the image.
logo = 'Pytorch_logo.png'
logo_img = cv2.imread(logo, cv2.IMREAD_COLOR)

# Print the size of the image.
print("Image size is ", logo_img.shape)

# Display the image.
plt.figure(figsize = (12, 12))
plt.imshow(logo_img);


"""What happened?
The color channels need to be swapped as in the previous example, 
but there is also a black background that was unexpected.

Use cv2.IMREAD_UNCHANGED to read the image with the alpha channel
PNG images support a 4th channel called the "alpha" channel. 
The alpha channel contains transparency information that allows specific regions within an image 
to appear transparent. As an example, consider the Facebook logo in the previous section. The 
logo contains two colors (blue and white). The white letters in the logo are actually 
white: (255, 255, 255). The PyTorch logo, on the other hand, contains an alpha channel that allows certain regions
of the image to appear transparent. So the "white" background is not white. Instead, those pixels 
are being masked by a 4th (alpha) channel, and are interpreted as transparent. In this case 
the pixels in the background portion of the image are set to:(0,0,0), which will appear as black unless we 
include the alpha chanel to mask them. We will cover transparency and alpha masking in a 
future module in more detail, but it is important to be aware of these details when reading and displaying images."""


# Read the image with the alpha channel.
logo_img = cv2.imread(logo, cv2.IMREAD_UNCHANGED)
logo_img =cv2.cvrtColor(logo_img, cv2.BGRA2RGBA)

plt.figure(figsize = [12, 12])
plt.imshow(logo_img);