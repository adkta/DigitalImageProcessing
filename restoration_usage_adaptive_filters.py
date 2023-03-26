import matplotlib.pyplot as plt
from matplotlib import rc
from skimage import io, img_as_ubyte, img_as_float64
from skimage.util import random_noise
from spatial import nonlinear_filter, averaging_filter
from restoration import geometric_mean, adaptive_mean_filter, adaptive_mean_filter_ver2, adaptive_median_filter, normalize
import numpy as np



rc('image', cmap='gray')
font = {'size': 7}
rc('font', **font)

########################
#ADAPTIVE MEAN FILTERING
########################
img = io.imread("/Users/newuser/COURSE MATERIALS/078MSICE/SEM III/Digital Image Processing/Assignment/Images/Input_PCB.png", as_gray=True)
img = img_as_ubyte(img)
noise = np.random.normal(loc=0, scale=30, size=img.shape)
noisy_img = img + noise
noisy_img = normalize(noisy_img, max_val=1, integer_val=False)
fig, axes = plt.subplots(2,2)
axes[0,0].imshow(noisy_img)
axes[0,0].title.set_text("Corrupted by Gaussian Noise")
restored_img = averaging_filter(img=noisy_img, window_size=7)
axes[0,1].imshow(restored_img)
axes[0,1].title.set_text("7x7 Arithmetic Mean Filter")
# noisy_img = img_as_ubyte(noisy_img)
# restored_img = geometric_mean(img=noisy_img, window_size=7)
# axes[1,0].imshow(restored_img)
# axes[1,0].title.set_text("7x7 Geometric Mean Filter")
# restored_img = adaptive_mean_filter(img = noisy_img, noise_var=30, window_size= 7)
# axes[1,1].imshow(restored_img)
# axes[1,1].title.set_text("7x7 Adaptive Mean Filter")
plt.show()

##########################
#ADAPTIVE MEDIAN FILTERING
##########################
# img = io.imread("/Users/newuser/COURSE MATERIALS/078MSICE/SEM III/Digital Image Processing/Assignment/Images/Input_PCB.png", as_gray=True)
# img = img_as_ubyte(img)
# noisy_img = random_noise(image=img, mode='s&p', amount=0.1)
# fig, axes = plt.subplots(2,2)
# axes[0,0].imshow(img)
# axes[0,0].title.set_text("Input Img")
# axes[0,1].imshow(noisy_img)
# axes[0,1].title.set_text("Corrupted by Salt and Pepper Noise")
# restored_img = nonlinear_filter(img=noisy_img, window_size=7, type='median')
# axes[1,0].imshow(restored_img)
# axes[1,0].title.set_text("7x7 Median Filter")
# restored_img = adaptive_median_filter(img  = noisy_img, max_window_size= 7)
# axes[1,1].imshow(restored_img)
# axes[1,1].title.set_text("Adaptive Median Filter (max = 7x7)")
# plt.show()


