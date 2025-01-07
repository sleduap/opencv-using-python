import cv2
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
# Derfine a kernel size.
kernel_size = 5
# Create a 5*5 kernel with all elements equal to 1.
kernel = np.ones((kernel_size, kernel_size), dtype = np.float32) / kernel_size**2

print (kernel)


# Perform convolution on an image.
filename = 'kitten.jpg'
image = cv2.imread(filename)

dst = cv2.filter2D(image, ddepth = -1, kernel = kernel)

plt.figure(figsize = [20,10])
plt.subplot(121); plt.axis('off'); plt.imshow(image[:,:,::-1]); plt.title("Original Image")
plt.subplot(122); plt.axis('off'); plt.imshow(dst[:,:,::-1]);   plt.title("Convolution Result");

# Box Blurring
# Apply a box filter - kernel size 5.
box_blur1 = cv2.blur(image, (5,5))

# Apply a box filter - kernel size 11.
box_blur2 = cv2.blur(image, (11,11))

# Display.
plt.figure(figsize = (20, 10))
plt.subplot(121); plt.axis('off'); plt.imshow(box_blur1[:,:,::-1]); plt.title('Blur 5x5 kernel')
plt.subplot(122); plt.axis('off'); plt.imshow(box_blur2[:,:,::-1]); plt.title('Blur 11x11 kernel');

# Gaussian Blurring
# Apply Gaussian blur.
gaussian_blur1 = cv2.GaussianBlur(image, (5,5), 0, 0)
gaussian_blur2 = cv2.GaussianBlur(image, (11,11), 0, 0)

# Display.
plt.figure(figsize = (20, 8))
plt.subplot(121); plt.axis('off'); plt.imshow(gaussian_blur1[:,:,::-1]); plt.title('Blur 5x5 kernel')
plt.subplot(122); plt.axis('off'); plt.imshow(gaussian_blur2[:,:,::-1]); plt.title('Blur 11x11 kernel');


# compare the results of box and Gaussian blurring.

plt.figure(figsize = (20, 8))
plt.subplot(121); plt.axis('off'); plt.imshow(box_blur2[:,:,::-1]);      plt.title('Box Blur 11x11 kernel')
plt.subplot(122); plt.axis('off'); plt.imshow(gaussian_blur2[:,:,::-1]); plt.title('Gaussian Blur 11x11 kernel');

#Gaussian Blur and effect of sigma 
# Specifying sigmax = 0 and sigmay = 0, will compute a sigma of 2 for a 11x11 kernal
gaussian_blur3 = cv2.GaussianBlur(image, (11,11), 0, 0) 
# Here we are explicity setting the sigma values to be very large.
gaussian_blur4 = cv2.GaussianBlur(image, (11,11), 5, 5)

# Display.
plt.figure(figsize = (20, 8))
plt.subplot(121); plt.axis('off'); plt.imshow(gaussian_blur3[:,:,::-1]); plt.title('Gaussian Blur, sigma = 2')
plt.subplot(122); plt.axis('off'); plt.imshow(gaussian_blur4[:,:,::-1]); plt.title('Gaussian Blur, sigma = 5');



#Image Sharpening
saturn = cv2.imread('saturn.jpg')

# Define a sharpening kernel.
kernel = np.array([[ 0, -1,  0],
                   [-1,  5, -1],
                   [ 0, -1,  0]])

saturn_sharp = cv2.filter2D(saturn, ddepth = -1, kernel = kernel)

plt.figure(figsize = (20, 15))
plt.subplot(121); plt.axis('off'); plt.imshow(saturn[:,:,::-1]);       plt.title('Telescope Image of a Saturn')
plt.subplot(122); plt.axis('off'); plt.imshow(saturn_sharp[:,:,::-1]); plt.title('Saturn Sharpened');

#Recovering sharpness from Gaussian blurred image
image = cv2.imread('kitten_zoom.png')

gaussian_blur = cv2.GaussianBlur(image, (11,11), 0, 0) 

# Sharpening kernel.
kernel1 = np.array([[ 0, -1,  0],
                   [-1,  5, -1],
                   [ 0, -1,  0]])

# More extreme sharpening kernel.
kernel2 = np.array([[0,  -4,  0],
                   [-4,  17, -4],
                   [ 0,  -4,  0]])

# Apply sharpening.
image_sharp1 = cv2.filter2D(gaussian_blur, ddepth = -1, kernel = kernel1)
image_sharp2 = cv2.filter2D(gaussian_blur, ddepth = -1, kernel = kernel2)

# Display.
plt.figure(figsize = (20,10))
plt.subplot(141); plt.axis('off'); plt.imshow(image[:,:,::-1]);         plt.title('Original')
plt.subplot(142); plt.axis('off'); plt.imshow(gaussian_blur[:,:,::-1]); plt.title('Gaussian Blur (11x11)')
plt.subplot(143); plt.axis('off'); plt.imshow(image_sharp1[:,:,::-1]);  plt.title('Sharpened')
plt.subplot(144); plt.axis('off'); plt.imshow(image_sharp2[:,:,::-1]);  plt.title('Sharpened More');