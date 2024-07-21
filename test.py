import cv2
import numpy as np
from matplotlib import pyplot as plt

# Create a larger sample binary image with steppy curves
binary_image = np.array([
    [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0],
    [0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0],
    [1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1],
    [1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1],
    [0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0],
    [0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0],
    [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0]
], dtype=np.uint8)

# Print original binary image
print("Original binary image:")
print(binary_image)

# Apply Gaussian blur to smooth the edges
blurred = cv2.GaussianBlur(binary_image.astype(np.float32), (3, 3), 0)

# Print smoothed image with decimal places
print("\nSmoothed image with decimal places:")
print(blurred)

# Normalize the blurred image for better visualization (optional)
normalized_blurred = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX)

# Display the images for comparison
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title('Original Binary Image')
plt.imshow(binary_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Smoothed Image with Decimal Places')
plt.imshow(normalized_blurred, cmap='gray')
plt.axis('off')

plt.show()
