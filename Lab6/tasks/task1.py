import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Read the image
image = cv2.imread('Lab6/image1.png', cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Error: Image not found.")
else:
    # 2. Initial Thresholding
    # Use a moderate threshold to get clean binary image
    _, thresh = cv2.threshold(image, 40, 255, cv2.THRESH_BINARY)

    # 3. Apply Gaussian Blur to connect broken parts
    # Kernel size must be odd number
    blurred = cv2.GaussianBlur(thresh, (49,49), 0)

    # 4. Morphological Closing to connect gaps
    # Use moderate kernel for good connection
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    closed = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel_close, iterations=1)

    # 5. Apply slight dilation to strengthen characters
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
    dilated = cv2.dilate(closed, kernel_dilate, iterations=1)

    # 6. Final Thresholding to clean edges
    _, final_result = cv2.threshold(dilated, 150, 255, cv2.THRESH_BINARY)

    # --- Display Results ---
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title("Original (Broken)")
    plt.imshow(image, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("After Closing")
    plt.imshow(dilated, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Final Result")
    plt.imshow(final_result, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # Save the output
    cv2.imwrite('Lab6/results/image1_repaired.png', final_result)
    
    print("Image processing completed!")
    print("\nAlgorithm Explanation:")
    print("=" * 60)
    print("1. Binary Thresholding (threshold=130):")
    print("   - Creates clean black and white image")
    print("   - Separates text from background")
    print("")
    print("2. Gaussian Blur (41x41 kernel):")
    print("   - Large blur smooths and connects nearby broken pixels")
    print("   - Spreads white pixels to bridge gaps")
    print("   - Kernel size must be odd number")
    print("")
    print("3. Morphological Closing (9x9 ellipse, 2 iterations):")
    print("   - Dilation followed by erosion")
    print("   - Connects gaps in broken characters")
    print("   - Ellipse kernel creates smooth, natural edges")
    print("   - Multiple iterations ensure complete connection")
    print("")
    print("4. Final Thresholding (threshold=85):")
    print("   - Converts blurred grayscale back to binary")
    print("   - Lower threshold captures blur-connected pixels")
    print("   - Ensures broken parts are included")
    print("")
    print("Why this approach works:")
    print("- Large Gaussian blur bridges gaps naturally")
    print("- Elliptical closing smooths edges")
    print("- Lower final threshold includes connected areas")
    print("- Result: Clean, connected, readable text")