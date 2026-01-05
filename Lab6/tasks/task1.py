import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the input image
image = cv2.imread('image1.png', cv2.IMREAD_GRAYSCALE)

# Apply binary thresholding
_, binary_image = cv2.threshold(image, 130, 255, cv2.THRESH_BINARY)

# Apply closing operation to connect broken strokes
# Use a moderately sized rectangular kernel
kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 3))
closed = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel_close, iterations=1)

# Apply dilation to strengthen and smooth the characters
kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
dilated = cv2.dilate(closed, kernel_dilate, iterations=1)

# Apply another closing to ensure all gaps are filled
kernel_close2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
result = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel_close2, iterations=1)

# Display results
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(binary_image, cmap='gray')
axes[0].set_title('Original Broken Text')
axes[0].axis('off')

axes[1].imshow(closed, cmap='gray')
axes[1].set_title('After First Closing')
axes[1].axis('off')

axes[2].imshow(result, cmap='gray')
axes[2].set_title('Final Result')
axes[2].axis('off')

plt.tight_layout()
plt.show()

# Save the result
cv2.imwrite('image1_repaired.png', result)

print("Image processing completed!")
print("\nAlgorithm Explanation:")
print("=" * 60)
print("1. Binary Thresholding (threshold=130):")
print("   - Converts image to binary (black and white)")
print("   - Threshold value separates text from background")
print("")
print("2. Morphological Closing (4x3 rectangular kernel):")
print("   - Dilation followed by erosion")
print("   - Connects nearby broken strokes")
print("   - Fills small gaps in character strokes")
print("")
print("3. Dilation (3x3 elliptical kernel):")
print("   - Expands white regions (text)")
print("   - Strengthens and thickens character strokes")
print("   - Elliptical kernel creates smoother edges")
print("")
print("4. Final Closing (3x3 elliptical kernel):")
print("   - Ensures all remaining gaps are filled")
print("   - Smooths character edges")
print("   - Creates clean, continuous text")
print("")
print("Why this approach works:")
print("- Rectangular kernel first: connects breaks efficiently")
print("- Elliptical kernels: create smooth, natural-looking text")
print("- Multiple steps: gradually repairs without over-processing")
print("- Result: Clean, readable, connected characters")