import numpy as np
from skimage import io, img_as_ubyte, exposure
import matplotlib as mpl
import matplotlib.pyplot as plt

from spatial import show_image, bit_plane_slicing, negative, averaging_filter, nonlinear_filter, high_pass_filter, \
    high_filter, laplacian, prewitt, sobel

#LAPLACIAN


#FIRST DERIVATIVE FILTERS
#PREWITT FILTER


# Matplotlib's default color map is viridis.
# Checkout this colormap at https://matplotlib.org/stable/tutorials/colors/colormaps.html
# This default colormap is set in the config file matplotlibrc located at matlplotlib.matplotlib_fname()
# If we write plt.imshow(img_gray) without specifying cmap, the image will be displayed in viridis
# To display in gray colour map, do one of the following:
#   1)Do plt.imshow(img_gray, cmap='gray') everytime .
#   2)Default colormap for this program: mpl.rc('image', cmap='gray') - RECOMMENDED
#   3)Change the config file matplotlibrc
mpl.rc('image', cmap='gray')
font = {'size': 5}
mpl.rc('font', **font)
OP_DIRECTORY = "/Users/newuser/PycharmProjects/DigitalImageProcessing/Output/Spatial/"
img_gray = io.imread("/Users/newuser/COURSE MATERIALS/078MSICE/SEM III/Digital Image Processing/Assignment/Images/NPR100_Bill.jpg", as_gray=True)

#Check shape to see if we've actually read in gray level format. Gray Level is normalized (levels between 0 and 1)
# print(img_gray.shape)
# print("Read in gray level data type: " , img_gray.dtype)
# print(img_gray)

# Convert from normalized to equivalent 8 bit integer.
# Not to be confused with img_as_uint. This will convert to 16 bit integer.
# img_gray_8bit = img_as_ubyte(img_gray)
# # print(img_gray_8bit)
#
# bit_plane_sliced_img_7 = bit_plane_slicing(img=img_gray_8bit, bit_plane=7)
# bit_plane_sliced_img_6 = bit_plane_slicing(img=img_gray_8bit, bit_plane=6)
# bit_plane_sliced_img_5 = bit_plane_slicing(img=img_gray_8bit, bit_plane=5)
# bit_plane_sliced_img_4 = bit_plane_slicing(img=img_gray_8bit, bit_plane=4)
# bit_plane_sliced_img_3 = bit_plane_slicing(img=img_gray_8bit, bit_plane=3)
# bit_plane_sliced_img_2 = bit_plane_slicing(img=img_gray_8bit, bit_plane=2)
# bit_plane_sliced_img_1 = bit_plane_slicing(img=img_gray_8bit, bit_plane=1)
# bit_plane_sliced_img_0 = bit_plane_slicing(img=img_gray_8bit, bit_plane=0)
#
# fig, axes = plt.subplots(3,3)
# axes[0,0].imshow(img_gray)
# axes[0,0].title.set_text("Original Image")
# axes[0,1].imshow(bit_plane_sliced_img_7)
# axes[0,1].title.set_text("Slice 7")
# axes[0,2].imshow(bit_plane_sliced_img_6)
# axes[0,2].title.set_text("Slice 6")
# axes[1,0].imshow(bit_plane_sliced_img_5)
# axes[1,0].title.set_text("Slice 5")
# axes[1,1].imshow(bit_plane_sliced_img_4)
# axes[1,1].title.set_text("Slice 4")
# axes[1,2].imshow(bit_plane_sliced_img_3)
# axes[1,2].title.set_text("Slice 3")
# axes[2,0].imshow(bit_plane_sliced_img_2)
# axes[2,0].title.set_text("Slice 2")
# axes[2,1].imshow(bit_plane_sliced_img_1)
# axes[2,1].title.set_text("Slice 1")
# axes[2,2].imshow(bit_plane_sliced_img_0)
# axes[2,2].title.set_text("Slice 0")
# plt.savefig(fname=OP_DIRECTORY+"Bitplane Slicing", dpi=300)
#
# # print(bit_plane_sliced_img_7)
# # show_image(bit_plane_sliced_img_7)
#
# negative_img = negative(img_gray)
# show_image(negative_img)
# fig, axes = plt.subplots(1,2)
# axes[0].imshow(img_gray_8bit)
# axes[0].title.set_text("Original")
# axes[1].imshow(negative_img)
# axes[1].title.set_text("Negative")
# plt.savefig(fname=OP_DIRECTORY+"Negative", dpi=300)
#
# # #Gamma and Log Adjustment
# overexposed_img_gray = io.imread("/Users/newuser/COURSE MATERIALS/078MSICE/SEM III/Digital Image Processing/Assignment/Images/Overexposed.jpg", as_gray=True)
# # # Don't need to convert to 8 bit integer as the gamma correction occurs on normalized values.
# gamma_value = 7
# gamma_adjusted_img = exposure.adjust_gamma(image=overexposed_img_gray, gamma = gamma_value, gain=1)
# log_adjusted_img = exposure.adjust_log(image=overexposed_img_gray, gain=1, inv=True)
# # #Comparing side by side
# fig = plt.figure(figsize=(10,10))
# # #1,2,1 = 1*2 matrix. last argument 1 = first image
# sub1 = fig.add_subplot(1,3,1)
# sub1.imshow(overexposed_img_gray)
# sub1.title.set_text("Overexposed")
# sub2 = fig.add_subplot(1,3,2)
# sub2.imshow(log_adjusted_img)
# sub2.title.set_text("Log Adjusted")
# sub3 = fig.add_subplot(1,3,3)
# sub3.imshow(gamma_adjusted_img)
# sub3.title.set_text(f"Gamma Adjusted (Gamma = {gamma_value})")
# plt.savefig(fname=OP_DIRECTORY+"Power Law Correcting OverExposure", dpi=300)
# plt.show()
#
#
# underexposed_img_gray = io.imread("/Users/newuser/COURSE MATERIALS/078MSICE/SEM III/Digital Image Processing/Assignment/Images/Underexposed.jpg", as_gray=True)
# log_corrected_img = exposure.adjust_log(image=underexposed_img_gray, gain=1, inv=False)
# gamma_value= 0.7
# gamma_corrected_img = exposure.adjust_gamma(image=underexposed_img_gray,gamma=gamma_value,gain=1)
# fig = plt.figure(figsize=(10,10))
# #1,2,1 = 1*2 matrix. last argument 1 = first image
# sub1 = fig.add_subplot(1,3,1)
# sub1.imshow(underexposed_img_gray)
# sub1.title.set_text("Underexposed")
# sub2 = fig.add_subplot(1,3,2)
# sub2.imshow(log_corrected_img)
# sub2.title.set_text("Log Corrected")
# sub3 = fig.add_subplot(1,3,3)
# sub3.imshow(gamma_corrected_img)
# sub3.title.set_text(f"Gamma Adjusted (Gamma = {gamma_value})")
# plt.savefig(fname=OP_DIRECTORY+"Power Law Correcting Underexposure", dpi=300)
# plt.show()
#
# #HISTOGRAM EQUALIZATION
# underexposed_img_gray = io.imread("/Users/newuser/COURSE MATERIALS/078MSICE/SEM III/Digital Image Processing/Assignment/Images/Underexposed.jpg", as_gray=True)
# hist_equalized_img = exposure.equalize_hist(underexposed_img_gray)
# fig = plt.figure(figsize = (10,10))
# sub1 = fig.add_subplot(2,2,1);
# sub1.imshow(underexposed_img_gray)
# sub1.title.set_text("UnderExposed")
# sub2 = fig.add_subplot(2,2,2);
# sub2.imshow(hist_equalized_img)
# sub2.title.set_text("Equalized")
# sub3 = fig.add_subplot(2,2,3);
# hist, bin_centers = exposure.histogram(underexposed_img_gray)
# sub3.plot(bin_centers, np.divide(hist, underexposed_img_gray.shape[0]*underexposed_img_gray.shape[1]))
# sub3.title.set_text("UnderExposed Histogram")
# sub4 = fig.add_subplot(2,2,4);
# hist, bin_centers = exposure.histogram(hist_equalized_img)
# sub4.plot(bin_centers, np.divide(hist, underexposed_img_gray.shape[0]*underexposed_img_gray.shape[1]))
# sub4.title.set_text("Equalized Histogram")
# plt.savefig(fname=OP_DIRECTORY+"Histogram Equalization", dpi = 300)
# plt.show()
#
# #SPATIAL FILTERING
#
# #DENOISING FILTER: MEDIAN FILTER, MAX FILTER AND MIN FILTER
# spnoise_10percent_img = io.imread("/Users/newuser/COURSE MATERIALS/078MSICE/SEM III/Digital Image Processing/Assignment/Images/Salt_Pepper_Noise_10percent.png", as_gray=True)
# spnoise_30percent_img = io.imread("/Users/newuser/COURSE MATERIALS/078MSICE/SEM III/Digital Image Processing/Assignment/Images/Salt_Pepper_Noise_30percent.png", as_gray=True)
# #
# median_filtered_img_spn_30 = nonlinear_filter(spnoise_30percent_img, 3, "median")
# median_filtered_img_spn_10 = nonlinear_filter(spnoise_10percent_img, 3, "median")
# f, axarr = plt.subplots(2,2)
# axarr[0,0].imshow(spnoise_10percent_img)
# axarr[0,0].title.set_text("10% Salt and Paper Noise")
# axarr[0,1].imshow(median_filtered_img_spn_10)
# axarr[0,1].title.set_text("Complete removal")
# axarr[1,0].imshow(spnoise_30percent_img)
# axarr[1,0].title.set_text("30% Salt and Paper Noise")
# axarr[1,1].imshow(median_filtered_img_spn_30)
# axarr[1,1].title.set_text("Incomplete removal")
# plt.savefig(fname=OP_DIRECTORY+"Median Filter", dpi = 300)
# plt.show()
#
# #SMOOTHING FILTER OR LP FILTER OR AVERAGING FILTER
# img = io.imread("/Users/newuser/COURSE MATERIALS/078MSICE/SEM III/Digital Image Processing/Assignment/Images/shapes_and_a.jpg", as_gray=True)
#
# avg_filter_3x3 = averaging_filter(img, 3)
# avg_filter_5x5 = averaging_filter(img, 5)
# f, axarr = plt.subplots(1,3)
# axarr[0].imshow(img)
# axarr[0].title.set_text("Original")
# axarr[1].imshow(avg_filter_3x3)
# axarr[1].title.set_text("3*3 Window Averaging")
# axarr[2].imshow(avg_filter_5x5)
# axarr[2].title.set_text("5*5 Window Averaging ")
# plt.savefig(fname=OP_DIRECTORY+"Averaging Filter", dpi = 300)
# plt.show()

#SHARPENING HIGH-PASS FILTERS

#HIGH-PASS FILTERS (SIMPLE, DERIVATIVE FILTERS (1st and 2nd)) HIGH BOOST AND HIGH FREQUENCY EMPHASIS FILTERS
house_img = io.imread("/Users/newuser/COURSE MATERIALS/078MSICE/SEM III/Digital Image Processing/Assignment/Images/simple_house.jpg", as_gray=True)
#simple high-pass filter (without specifying mask the function uses a simple default mask)
hp_filtered_img = high_pass_filter(img= house_img)
hb_filtered_img = high_filter(img=house_img, a=1.1)
hfe_filtered_img = high_filter(img=house_img, a=1.1, b=1.5, filter_type='High Frequency Emphasis')
laplacian_enhanced_img = laplacian(img=house_img)
prewitt_filtered_img = prewitt(img=house_img)
sobel_filtered_img = sobel(img=house_img)

f, axarr = plt.subplots(2,4)
axarr[0,0].imshow(house_img)
axarr[0,0].title.set_text("Original")
axarr[0,1].imshow(hp_filtered_img)
axarr[0,1].title.set_text("Simple HPF")
axarr[0,2].imshow(prewitt_filtered_img)
axarr[0,2].title.set_text("Prewitt")
axarr[0,3].imshow(sobel_filtered_img)
axarr[0,3].title.set_text("Sobel")
axarr[1,0].imshow(laplacian_enhanced_img)
axarr[1,0].title.set_text("Laplacian")
axarr[1,1].imshow(hb_filtered_img)
axarr[1,1].title.set_text("High Boost")
axarr[1,2].imshow(hfe_filtered_img)
axarr[1,2].title.set_text("HF Emphasis")
plt.axis('off')
plt.savefig(fname=OP_DIRECTORY+"High Pass, High Boost and High Frequency Emphasis Filters", dpi = 300)
plt.show()


# # Thresholding
# my_writing_img_two = io.imread("/Users/newuser/COURSE MATERIALS/078MSICE/SEM III/Digital Image Processing/Assignment/Images/FourierOfADerivative.JPG",as_gray=True)
# fig, axes = plt.subplots(1, 2)
# axes[0].imshow(my_writing_img_two)
# axes[0].title.set_text("Original")
# my_writing_img_two = img_as_ubyte(my_writing_img_two)
# with np.nditer(my_writing_img_two, op_flags=['readwrite']) as it:
#     for x in it:  # specifying op_flags as write also since we want to modify the image intensity ndarray
#         x[...] = 255 if x[...] > 95 else 0
#
# axes[1].imshow(my_writing_img_two)
# axes[1].title.set_text("Thresholded Image")
# plt.savefig(OP_DIRECTORY+"Thresholding", dpi = 300)
# plt.show()
