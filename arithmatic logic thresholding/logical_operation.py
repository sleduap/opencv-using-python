import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image 

plt.rcParams['image.cmap'] = 'gray'


img_rec = cv2.imread('rectangle.jpg', cv2.IMREAD_GRAYSCALE)
img_cir = cv2.imread('circle.jpg', cv2.IMREAD_GRAYSCALE)

plt.figure(figsize = [20,5])
plt.subplot(121);  plt.imshow(img_rec);
plt.subplot(122);  plt.imshow(img_cir);
print(img_rec.shape)

result = cv2.bitwise_and(img_rec, img_cir, mask = None)

plt.imshow(result);



result = cv2.bitwise_or(img_rec, img_cir, mask = None)
plt.imshow(result);


result = cv2.bitwise_xor(img_rec, img_cir, mask = None)
plt.imshow(result);



# Read the of image of color CR logo (foreground).
img_logo = cv2.imread('CR_Logo.png', cv2.IMREAD_COLOR)

# Print the image shape.
print(img_logo.shape)
logo_h = img_logo.shape[0]
logo_w = img_logo.shape[1]

# Display the image.
plt.figure(figsize = [5, 5])
plt.imshow(img_logo[:, :, ::-1]);



# Read the of image of color cheackerboad (background).
img_background = cv2.imread('checkerboard_color.png', cv2.IMREAD_COLOR)

# Print the image shape.
print(img_background.shape);




# Set the dimension of the background image to be the same as the logo.
dim = (logo_w, logo_h)

# Resize the background image to the same size as logo image.
img_background = cv2.resize(img_background, dim, interpolation = cv2.INTER_AREA)

# Print the image shape to confirm it's the same size as the logo.
print(img_background.shape)

# Display the image.
plt.figure(figsize = [5, 5])
plt.imshow(img_background[:, :, ::-1]);


# Set the dimension of the background image to be the same as the logo.
dim = (logo_w, logo_h)

# Resize the background image to the same size as logo image.
img_background = cv2.resize(img_background, dim, interpolation = cv2.INTER_AREA)

# Print the image shape to confirm it's the same size as the logo.
print(img_background.shape)

# Display the image.
plt.figure(figsize = [5, 5])
plt.imshow(img_background[:, :, ::-1]);



# Create colorful checkerboard background "behind" the logo lettering.
img_background = cv2.bitwise_and(img_background, img_background, mask = img_logo_mask)

# Print the image shape.
print(img_background.shape);

# Display the image.
plt.figure(figsize = [5, 5])
plt.imshow(img_background);


# Create an inverse mask.
img_logo_mask_inv = cv2.bitwise_not(img_logo_mask)

# Print the image shape.
print(img_logo_mask_inv.shape)

# Display the image.
plt.figure(figsize = [5, 5])
plt.imshow(img_logo_mask_inv);


# Isolate the foreground using the inverse mask.
img_foreground = cv2.bitwise_and(img_logo, img_logo, mask = img_logo_mask_inv)

# Print the image shape.
print(img_foreground.shape)

# Display the image.
plt.figure(figsize = [5, 5])
plt.imshow(img_foreground);



# Add the two previous results to obtain the final result.
result = cv2.add(img_background,img_foreground)

# Display the image and save the the result to the file system.
plt.figure(figsize = [5, 5])
plt.imshow(result[:, :, ::-1])
cv2.imwrite('logo_final.png', result);



