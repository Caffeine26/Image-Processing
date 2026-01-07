import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


filename = 'coin.png'
img_color = cv2.imread(filename)
if img_color is None:
    print(f"ERROR: Cannot load '{filename}'. Check the filename and path!")
else:

    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    

    _, binary = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)

  
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))


    eroded = cv2.erode(binary, kernel, iterations=1)
    boundary = cv2.subtract(binary, eroded)

    # Show result
    plt.figure(figsize=(10, 8))
    plt.imshow(boundary, cmap='gray')
    plt.title('Morphological Boundary')
    plt.axis('off')
    plt.show()

    # Save result
    os.makedirs('Lab6/results', exist_ok=True)
    cv2.imwrite('Lab6/results/boundary.png', boundary)
    print(" Boundary image saved at 'Lab6/results/boundary.png'")
