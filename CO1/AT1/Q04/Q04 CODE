# Unique Python Program
# Pixel Resolution vs Intensity Resolution Demonstration
# Runs in Google Colab
import cv2
import numpy as np
import matplotlib.pyplot as plt
# Create a synthetic medical-like grayscale image
img = np.zeros((256, 256), dtype=np.uint8)
# Draw circles representing organs/tissues
cv2.circle(img, (128, 128), 80, 180, -1)
cv2.circle(img, (128, 128), 40, 230, -1)
cv2.rectangle(img, (30, 180), (90, 240), 120, -1)
# -------- Pixel Resolution -------
low_pixel = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
pixel_result = cv2.resize(low_pixel, (256, 256), interpolation=cv2.INTER_NEAREST)
# -------- Intensity Resolution -------
levels = 16
intensity_result = (img // (256 // levels)) * (256 // levels)
# Display Results
titles = [
"Original Image",
"Reduced Pixel Resolution",
"Reduced Intensity Resolution"
]
images = [img, pixel_result, intensity_result]
plt.figure(figsize=(12,4))
for i in range(3):
plt.subplot(1,3,i+1)
plt.imshow(images[i], cmap="gray")
plt.title(titles[i])
plt.axis("off")
plt.show()
print("Observation:")
print("1. Lower pixel resolution decreases image sharpness.")
print("2. Lower intensity resolution decreases image contrast.")
print("3. In medical imaging, both high pixel and high intensity resolution improve diagnosis.")
