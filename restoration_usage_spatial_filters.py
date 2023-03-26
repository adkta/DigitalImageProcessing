from skimage import io, img_as_ubyte, img_as_float32
from skimage.util import random_noise
import numpy as np
from matplotlib import pyplot as plt, rc
from spatial import averaging_filter, show_image, nonlinear_filter
from restoration import geometric_mean, normalize, handle_intensity_ndarray, contraharmonic_mean_filter, alpha_trimmed_mean_filter

rc('image', cmap='gray')
font = {'size': 5}
rc('font', **font)

##############################
#ARITHMETIC VS GEOMETRIC MEAN
##############################
# img = io.imread("/Users/newuser/COURSE MATERIALS/078MSICE/SEM III/Digital Image Processing/Assignment/Images/shapes_and_a.jpg", as_gray=True)
# img = img_as_ubyte(img)
# fig, axes = plt.subplots(2,2)
# axes[0, 0].imshow(img, vmin = 0, vmax = 255)
# axes[0, 0].title.set_text("Input Image")
# #Add Gaussian Noise
# noise = np.random.normal(loc=127, scale=30, size=img.shape)
# noisy_img = img + noise
# noisy_img = normalize_to_8bit_integer(noisy_img)
# axes[0, 1].imshow(noisy_img, vmin = 0 , vmax = 255)
# axes[0, 1].title.set_text("Additive Gaussian Noise")
#
# #Arithmetic Filter
# normalized_noisy_img = noisy_img/255
# restored_img = averaging_filter(normalized_noisy_img, window_size=3)
# axes[1, 0].imshow(restored_img) # spatial.averaging_filter uses 0 to 1 levels
# axes[1, 0].title.set_text("3x3 Arithmetic Mean")
#
# #Geometric Filter
# restored_img = handle_intensity_ndarray(geometric_mean(noisy_img, 3))
# axes[1, 1].imshow(restored_img, vmin = 0 , vmax = 255)
# axes[1, 1].title.set_text("3x3 Geometric Mean")
#
# plt.show()


#######################
#CONTRAHARMONIC MEAN
#######################
# # img = io.imread("/Users/newuser/COURSE MATERIALS/078MSICE/SEM III/Digital Image Processing/Assignment/Images/pepper_noise.jpg", as_gray=True)
# # img = img_as_ubyte(img)
# # q= 1.5
# img = io.imread("/Users/newuser/COURSE MATERIALS/078MSICE/SEM III/Digital Image Processing/Assignment/Images/salt_noise.jpg", as_gray=True)
# img = img_as_ubyte(img)
# q= -5
# restored_img = contraharmonic_mean_filter(img = img, window_size=3, q = q)
# fig, axes = plt.subplots(1, 2)
# axes[0].imshow(img)
# axes[0].title.set_text("Salt Noise")
# axes[1].imshow(restored_img)
# axes[1].title.set_text(f"Contraharmonic Mean Filter Q = {q}")
# plt.show()


#################
#COMPARISON - Mean, median and alpha trimmed mean
img = io.imread("/Users/newuser/COURSE MATERIALS/078MSICE/SEM III/Digital Image Processing/Assignment/Images/Input_PCB.png", as_gray=True)
noise = np.random.uniform(size=img.shape)
noisy_img = img + noise
noisy_img = normalize(noisy_img, max_val=1, integer_val=False)  #Additive Filter was poorly coded so need to normalize between 0 and 1
# fig, axes = plt.subplots(1, 2)
# axes[0].imshow(noisy_img)
# axes[0].title.set_text("Additive Uniform Noise")
noisy_img = random_noise(noisy_img, mode='s&p',amount=0.1)
# axes[1].imshow(noisy_img)
# axes[1].title.set_text("Salt and Pepper Noise")
window_size = 5
# restored_img = averaging_filter(img=noisy_img, window_size=window_size) #Arithmetic Filter takes intensities from 0 to 1
# axes[0].imshow(restored_img)
# axes[0].title.set_text(f"{window_size}x{window_size} Arithmetic Mean Filter")
noisy_img= img_as_ubyte(noisy_img)#Converting to 8 bit integer because float operations are more expensive
# restored_img = nonlinear_filter(img=noisy_img, window_size=window_size, type="median")
# axes[1].imshow(restored_img, vmin = 0, vmax = 255)
# axes[1].title.set_text(f"{window_size}x{window_size} Median Filter")
# restored_img = geometric_mean(img=noisy_img, window_size=window_size)
# axes[1,1].imshow(restored_img)
# axes[1,1].title.set_text(f"{window_size}x{window_size} Geometric Mean Filter")
d=6
restored_img = alpha_trimmed_mean_filter(img=noisy_img, window_size=window_size, d=d)
plt.imshow(restored_img, vmin=0, vmax=255)
plt.title(f"{window_size}x{window_size} Alpha Trimmed Mean Filter(d={d})")
# axes[2,1].imshow(restored_img, vmin=0, vmax=255)
# axes[2,1].title.set_text(f"{window_size}x{window_size} Alpha Trimmed Mean Filter(d={d})")
plt.show()