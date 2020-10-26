import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
import em_gaussian_mixture as em_gm

# read image
image = misc.imread('image/im.jpg', mode='RGB')
image = image.astype(float)/255
height, width, channels = image.shape

# train data
x = image.reshape(height*width, channels)

print("Insert value of K: ")
K = input()
max_iter = 300

# em algorithm
EM = em_gm.EM(x, int(K))
m, g = EM.ml_em(max_iter)

# image after segmentation
new_image = np.copy(x)
for pixel in range(x.shape[0]):
    new_image[pixel, :] = m[np.argmax(g[pixel])]

# reconstruction error
error = np.sum(np.square(new_image - x)) / x.shape[0]
print("Reconstruction error:", error)

new_image = new_image.reshape((height, width, channels))
plt.title("K=" + str(K) + "\nError=" + "%.2f" % (error*100) + "%")
plt.imshow(new_image)
plt.show()
