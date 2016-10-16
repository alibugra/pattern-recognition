import matplotlib.image as mpimg
import skimage.io as io
import numpy as np

from PIL import Image
from numpy import linalg as LA
from matplotlib import pyplot as plt

def plot_scatter(img):
    pixels = img.shape[0] * img.shape[1]
    channels = 3
    data = np.reshape(img[:, :, :channels], (pixels, channels))
    histo_rgb, _ = np.histogramdd(data, bins=256)
    r, g, b = np.nonzero(histo_rgb)
    # Calculate sample mean and covariance of the image pixels in r-g
    print np.mean(r)
    print np.mean(g)
    print np.cov(r, g)
    w, v = LA.eig(np.cov(r, g)) # Calculate eigenvectors of covariance matrix
    print w
    print v
    plt.scatter(r, g) # Show scatter matrix in r-g space
    plt.show()

# A color picture which is about face is loaded
image_file = io.imread('img/alibugra.jpg')
io.imshow(image_file)
io.show()

# Show the scatter matrix of the whole image in r-g space
plot_scatter(image_file)

# RGB image obtained
im = Image.open('img/alibugra.jpg')
rgb_im = im.convert('RGB')
r, g, b = rgb_im.getpixel((1, 1))
result_r = float(r) / (float(r) + float(g) + float(b)) # r = R / (R + G + B)
result_g = float(g) / (float(r) + float(g) + float(b)) # g = G / (R + G + B)

# Created mask and show it
im = Image.open('img/alibugra.jpg')
im = im.convert('RGBA')
data = np.array(im)
r1, g1, b1 = 250, 250, 250
rw, gw, bw, aw = 255, 255, 255, 255
red, green, blue, alpha = data[:,:,0], data[:,:,1], data[:,:,2], data[:,:,3]
mask = ((red > r1) & (green > g1))
io.imshow(mask)
io.show()
data[:,:,:4][mask] = [rw, gw, bw, aw]

# Plot the scatter matrix of face pixels
plot_scatter(data)
io.imshow(data)
io.show()
