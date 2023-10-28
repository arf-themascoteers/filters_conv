import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image = mpimg.imread("4.png")
plt.imshow(image, cmap='gray')
plt.show()