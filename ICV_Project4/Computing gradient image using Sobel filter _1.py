import cv2
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

image = cv2.imread(r'C:\Users\CENOTech\PycharmProjects\ICV_Project1\data\Lena.png', 0)

dx = cv2.Sobel(image, cv2.CV_32F, 1, 0)
dy = cv2.Sobel(image, cv2.CV_32F, 0, 1)

plt.figure(figsize=(8, 3))
plt.subplot(131)
plt.axis('off')
plt.title('image')
plt.imshow(image, cmap='gray')
plt.subplot(132)
plt.axis('off')
plt.imshow(dx, cmap='gray')
plt.title(r'$\frac{dI}{dx}$')
plt.imshow(dy, cmap='gray')
plt.tight_layout()
plt.show()
