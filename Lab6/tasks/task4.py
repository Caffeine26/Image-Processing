import cv2
import numpy as np
import matplotlib.pyplot as plt

# CHANGE THIS TO YOUR EXACT FILENAME
filename = 'coin.png' 

img_color = cv2.imread(filename)
if img_color is None:
    print(f"ERROR: Cannot load '{filename}'. Check the filename and path!")
else:
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    # Light blur to reduce noise while keeping edges sharp
    blurred = cv2.GaussianBlur(img_gray, (9, 9), 2)

    # Use Hough Circle Transform – perfect for detecting round coins like in your example
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,          # Inverse ratio of accumulator resolution
        minDist=50,      # Minimum distance between coin centers (adjust if coins are closer)
        param1=100,      # Canny edge threshold
        param2=35,       # Accumulator threshold – lower = more detections (but more false positives)
        minRadius=20,    # Adjust based on your coin sizes
        maxRadius=120
    )

    display_img = img_color.copy()
    coin_count = 0

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        
        for (x, y, r) in circles:
            # Draw green outer circle (like in your example image)
            cv2.circle(display_img, (x, y), r, (0, 255, 0), 4)
            
            # Draw small center point (optional)
            cv2.circle(display_img, (x, y), 2, (0, 0, 255), 3)
            
            coin_count += 1
            
            # Label the coin number near the center
            cv2.putText(display_img, str(coin_count), (x - 20, y - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    print(f"🪙 Coins detected with Hough Circle Transform: {coin_count}")

    # Show result
    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB))
    plt.title(f'Coins Detected: {coin_count}')
    plt.axis('off')
    plt.show()