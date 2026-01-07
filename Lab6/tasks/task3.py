import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the fingerprint image
image_path = 'Lab6/image3.png'  # Adjust if necessary
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError("Cannot load image3.png - check path")

# Step 1: Binarization - use a manual threshold or adjust Otsu if needed
# Otsu alone may not be perfect; try manual if Otsu over/under thresholds
_, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# If ridges are white in binary, invert to make ridges black
if np.mean(binary) > 128:  # if more white than black
    binary = cv2.bitwise_not(binary)

# Step 2: Remove small noise with opening
kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open, iterations=2)

# Step 3: Fill small gaps/holes with closing (use small kernel to avoid over-filling)
kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close, iterations=3)

# Step 4: Smooth edges with light opening
kernel_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
clean_fingerprint = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_smooth, iterations=1)

# Save the result
cv2.imwrite('image3_out.png', clean_fingerprint)

# Display
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.title('Original')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(binary, cmap='gray')
plt.title('Binary (Ridges Black)')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(clean_fingerprint, cmap='gray')
plt.title('Cleaned Fingerprint')
plt.axis('off')

plt.tight_layout()
plt.show()