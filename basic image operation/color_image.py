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

#################################################
#Split and Merging Color Channels
#################################################
img_bgr = cv2.imread('Emerald_Lakes_New_Zealand.jpg', cv2.IMREAD_COLOR)
b, g, r = cv2.split(img_bgr)

plt.figure(figsize = [20, 10])
plt.subplot(141); plt.imshow(r); plt.title('Red Channel')
plt.subplot(142); plt.imshow(g); plt.title('Green Channel')
plt.subplot(143); plt.imshow(b); plt.title('Blue Channel')

# Merge the individual channels into a BGR image.
imgMerged = cv2.merge((r, g, b))

# Display the merged output.
plt.subplot(144)
plt.imshow(imgMerged)
plt.title('Merged Output');



img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

# Split the image into the B,G,R components.
h, s, v = cv2.split(img_hsv)

# Display the channels.
plt.figure(figsize = [20, 5])
plt.subplot(141); plt.imshow(h); plt.title('H Channel')
plt.subplot(142); plt.imshow(s); plt.title('S Channel')
plt.subplot(143); plt.imshow(v); plt.title('V Channel')

# Display the original image.
plt.subplot(144); plt.imshow(img_bgr[:, :, ::-1]); plt.title('Original');




h_new = h + 10
img_hsv_merged = cv2.merge((h_new, s, v))
img_rgb_merged = cv2.cvtColor(img_hsv_merged, cv2.COLOR_HSV2RGB)

# Display the channels.
plt.figure(figsize = [20,5])
plt.subplot(141); plt.imshow(h_new); plt.title('H Channel')
plt.subplot(142); plt.imshow(s); plt.title('S Channel')
plt.subplot(143); plt.imshow(v); plt.title('V Channel')

# Display the modified image.
plt.subplot(144); plt.imshow(img_rgb_merged); plt.title('Modified');