from restoration import get_butterworth_band_reject_filter
from frequency import perform_Freq_Domain_Transform, fourier_spectrum, multipy_minus_one_power_x_plus_y
from skimage import io
from matplotlib import pyplot as plt
from matplotlib import rc


rc('image', cmap='gray')
font = {'size': 7}
rc('font', **font)


img = io.imread("/Users/newuser/COURSE MATERIALS/078MSICE/SEM III/Digital Image Processing/Assignment/Images/periodic_noise.png", as_gray=True)
h_uv = get_butterworth_band_reject_filter(img=img, d0=60, W=7, order=2)
restored_img = perform_Freq_Domain_Transform(img, h_uv)
fig, axes = plt.subplots(2,2)
axes[0,0].imshow(img)
axes[0,0].title.set_text("Periodic Noise")
axes[0,1].imshow(fourier_spectrum(multipy_minus_one_power_x_plus_y(img = img)))
axes[0,1].title.set_text("Input Image Fourier Spectrum")
axes[1,0].imshow(h_uv)
axes[1,0].title.set_text("Butterworth Band Reject Filter Spectrum")
axes[1,1].imshow(restored_img)
axes[1,1].title.set_text("Butterworth Filtered")
plt.show()
