import cv2
import matplotlib.pyplot as plt
# %matplotlib inline  
from IPython.display import Image
plt.rcParams['figure.figsize'] = (6.0, 6.0)
plt.rcParams['image.cmap'] = 'gray'



# Read the image.
sig_org = cv2.imread('signature.jpg', cv2.IMREAD_COLOR)

# Display the actual image in the browser.
Image(filename='signature.jpg', width = '400')


# Display the image using imshow() so we can see the size with axis.
plt.imshow(sig_org[:, :, ::-1])
plt.title('Sample Signature');

# Crop the signature from the original image.
sig = sig_org[700:2100, 450:3500, :]
plt.imshow(sig[:, :, ::-1]);

sig_gray = cv2.cvtColor(sig, cv2.COLOR_BGR2GRAY)
plt.imshow(sig_gray)
plt.title('Gray scale Sign');

ret, alpha_mask = cv2.threshold(sig_gray, 150, 255, cv2.THRESH_BINARY_INV)
cv2.imwrite('alpha_mask.jpg', alpha_mask)
plt.imshow(alpha_mask)
plt.title('Alpha Mask');

blue_mask = sig.copy()
blue_mask[:, :] = (255, 0, 0)
plt.imshow(blue_mask[:, :, ::-1]);


sig_color = cv2.addWeighted(sig, 1, blue_mask, 0.6, 0)
plt.imshow(sig_color[:, :, ::-1]);


# Split the color channels from the color image.
b, g, r = cv2.split(sig_color)
print(b.shape)
print(g.shape)
print(r.shape)



# Create a list of the four arrays with the alpha channel as the 4th member. These are four separate 2D arrays.
new = [b, g, r, alpha_mask]

# Use the merge() function to create a single, multi-channel array.
png = cv2.merge(new, 4)

# Save the transparent signature a PNG file to retain the alpha channel.
cv2.imwrite('extracted_sig.png', png)

# Display the actual image in the browser.
Image('extracted_sig.png', width = '400')