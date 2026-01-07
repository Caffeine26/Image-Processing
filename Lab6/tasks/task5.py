import cv2
import numpy as np
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
import os

# Create folder to save results
os.makedirs("Lab6/results", exist_ok=True)

# Read original image
img = cv2.imread("Lab6/leaf.png")
if img is None:
    raise FileNotFoundError("Cannot load leaf.png")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Step 1: Grayscale + Contrast Enhancement
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
gray_enhanced = clahe.apply(gray)
cv2.imwrite("Lab6/results/leaf/leaf_gray_enhanced.png", gray_enhanced)

# Step 2: Gaussian Blur
gray_blur = cv2.GaussianBlur(gray_enhanced, (5,5), 0)
cv2.imwrite("Lab6/results/leaf/leaf_gray_blur.png", gray_blur)

# Step 3: Adaptive Thresholding
binary = cv2.adaptiveThreshold(
    gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
    cv2.THRESH_BINARY_INV, 21, 10
)
cv2.imwrite("Lab6/results/leaf/leaf_binary.png", binary)

# Step 4: Morphological Cleaning
kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
binary_closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)

kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
binary_clean = cv2.morphologyEx(binary_closed, cv2.MORPH_OPEN, kernel_open)
cv2.imwrite("Lab6/results/leaf/leaf_binary_cleaned.png", binary_clean)

# Step 5: Optional Dilation
kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
binary_for_skel = cv2.dilate(binary_clean, kernel_dilate, iterations=1)

# Step 6: Skeletonization
binary_bool = binary_for_skel.astype(bool)
skeleton_thin = skeletonize(binary_bool)
skeleton = (skeleton_thin * 255).astype(np.uint8)
cv2.imwrite("Lab6/results/leaf/leaf_skeleton.png", skeleton)

# Display all steps
plt.figure(figsize=(20,6))

plt.subplot(1,5,1)
plt.title("Original Leaf")
plt.imshow(img_rgb)
plt.axis("off")

plt.subplot(1,5,2)
plt.title("Gray Enhanced")
plt.imshow(gray_enhanced, cmap="gray")
plt.axis("off")

plt.subplot(1,5,3)
plt.title("Binary")
plt.imshow(binary, cmap="gray")
plt.axis("off")

plt.subplot(1,5,4)
plt.title("Cleaned Binary")
plt.imshow(binary_clean, cmap="gray")
plt.axis("off")

plt.subplot(1,5,5)
plt.title("Skeleton")
plt.imshow(skeleton, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()

print("All images saved in Lab6/results folder.")
