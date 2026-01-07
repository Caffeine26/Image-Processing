import cv2
import numpy as np
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt

# Read original image (assuming "leaf.png" is in the working directory)
img = cv2.imread("leaf.png")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Step 1: Enhance contrast to bring out faint veins
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
gray_enhanced = clahe.apply(gray)

# Step 2: Slight Gaussian blur to reduce high-frequency noise
gray_blur = cv2.GaussianBlur(gray_enhanced, (5,5), 0)

# Step 3: Adaptive thresholding with larger block size for smoother thresholding
binary = cv2.adaptiveThreshold(
    gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
    cv2.THRESH_BINARY_INV, 21, 10  # Increased blockSize to 21, C to 10
)

# Step 4: Light morphological cleaning
# Small closing to fill tiny holes/gaps in veins
kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
binary_closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)

# Optional: very small opening to remove isolated pepper noise
kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
binary_clean = cv2.morphologyEx(binary_closed, cv2.MORPH_OPEN, kernel_open)

# Step 5: Optional single dilation to help connect nearby thin vein fragments
kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
binary_for_skel = cv2.dilate(binary_clean, kernel_dilate, iterations=1)

# Step 6: Skeletonization using scikit-image (better for fine details)
# Convert to boolean array as required by skimage.skeletonize
binary_bool = binary_for_skel.astype(bool)
skeleton_thin = skeletonize(binary_bool)

# Convert back to uint8 for display
skeleton = (skeleton_thin * 255).astype(np.uint8)

# Plot results
plt.figure(figsize=(20,6))

plt.subplot(1,5,1)
plt.title("Original Leaf Image")
plt.imshow(img_rgb)
plt.axis("off")

plt.subplot(1,5,2)
plt.title("Contrast Enhanced Grayscale")
plt.imshow(gray_enhanced, cmap="gray")
plt.axis("off")

plt.subplot(1,5,3)
plt.title("Binary Leaf Image")
plt.imshow(binary, cmap="gray")
plt.axis("off")

plt.subplot(1,5,4)
plt.title("Cleaned Binary Leaf")
plt.imshow(binary_clean, cmap="gray")
plt.axis("off")

plt.subplot(1,5,5)
plt.title("Improved Leaf Skeleton (Vein Structure)")
plt.imshow(skeleton, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()