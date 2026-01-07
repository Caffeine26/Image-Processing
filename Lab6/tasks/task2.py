import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Adjust path to your image
image_path = 'Lab6/image2.png'

if not os.path.exists(image_path):
    print(f"Error: Image not found at {image_path}")
    exit()

# Load grayscale
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    print("Error loading image")
    exit()

# Binarize
_, binary = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Use erosion to remove the thin connecting pixels in the middle
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
eroded = cv2.erode(binary, kernel, iterations=3)

dilated = cv2.dilate(eroded, kernel, iterations=9)


result = dilated   

# Save and display
cv2.imwrite('Lab6/results/cleaned_separated.png', result)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(binary, cmap='gray')
plt.title('Original binary')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(result, cmap='gray')
plt.title('After removing middle connection')
plt.axis('off')

plt.tight_layout()
plt.show()