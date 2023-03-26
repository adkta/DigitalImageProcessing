from skimage import exposure
from frequency import distance
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
from restoration import normalize
from scipy.stats import erlang

#Change colour map from viridis to gray
mpl.rc('image', cmap='gray')
font = {'size': 5}
mpl.rc('font', **font)

#Adding Gaussian Noise
gen_img = np.empty(shape=(400,400))
m, n = gen_img.shape
minimum_dim = min(m,n)
for x in range(m):
    for y in range(n):
        if distance(x, y, m/2, n/2) < minimum_dim/2:
            gen_img[x][y]= 220
        else:
            gen_img[x][y]=120
hist, bin_centers = exposure.histogram(gen_img)
fig, axes = plt.subplots(2,6)
axes[0,0].imshow(gen_img, vmin = 0, vmax = 255)
axes[1,0].plot(bin_centers, hist)
axes[1,0].title.set_text("Input Image")

# Adding Gaussian Noise
noise = np.random.normal(loc=127, scale=30, size=gen_img.shape)
noisy_img = gen_img + noise
noisy_img = normalize(noisy_img)
hist, bin_centers = exposure.histogram(noisy_img)
axes[0,1].imshow(noisy_img, vmin = 0, vmax = 255)
axes[1,1].plot(bin_centers, hist)
axes[1,1].title.set_text("Additive Gaussian Noise")

# Adding Rayleigh Noise
noise = np.random.rayleigh(scale=30, size=gen_img.shape)
noisy_img = gen_img + noise
noisy_img = normalize(noisy_img)
hist, bin_centers = exposure.histogram(noisy_img)
axes[0,2].imshow(noisy_img, vmin = 0, vmax = 255)
axes[1,2].plot(bin_centers, hist)
axes[1,2].title.set_text("Additive Rayleigh Noise")

# Adding Uniform Noise
noise = np.random.uniform(low=20,high=100,size=gen_img.shape)
noisy_img = gen_img + noise
noisy_img = normalize(noisy_img)
hist, bin_centers = exposure.histogram(noisy_img)
axes[0,3].imshow(noisy_img, vmin = 0, vmax = 255)
axes[1,3].plot(bin_centers, hist)
axes[1,3].title.set_text("Additive Uniform Noise")

# Adding Exponential Noise
noise = np.random.exponential(scale = 20, size = gen_img.shape)
noisy_img = gen_img + noise
noisy_img = normalize(noisy_img)
hist, bin_centers = exposure.histogram(noisy_img)
axes[0,4].imshow(noisy_img, vmin = 0, vmax = 255)
axes[1,4].plot(bin_centers, hist)
axes[1,4].title.set_text("Additive Exponential Noise")

# Adding Erlang Noise
noise = erlang.rvs(a = 1, loc=5, scale=20, size = gen_img.shape, random_state=None)
noisy_img = gen_img + noise
noisy_img = normalize(noisy_img)
print(noisy_img)
hist_noisy, bin_centers_noisy = exposure.histogram(noisy_img)
hist, bin_centers = exposure.histogram(noisy_img)
axes[0,5].imshow(noisy_img, vmin = 0, vmax = 255)
axes[1,5].plot(bin_centers, hist)
axes[1,5].title.set_text("Additive Erlang Noise")

plt.show()