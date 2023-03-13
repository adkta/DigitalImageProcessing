import numpy
import numpy as np
from matplotlib import pyplot as plot
import matplotlib as mpl
from skimage import io, filters

from frequency import fourier_spectrum, multipy_minus_one_power_x_plus_y, perform_Freq_Domain_Transform, \
    get_Ideal_LPF_FrequencyDomain, get_Ideal_HPF_FrequencyDomain, get_butterworth_lpf_frequency_domain, \
    get_butterworth_hpf_frequency_domain, get_gaussian_lpf_frequency_domain, get_gaussian_hpf_frequency_domain

#Change colour map from viridis to gray
mpl.rc('image', cmap='gray')
font = {'size': 5}
mpl.rc('font', **font)

OP_DIRECTORY = "/Users/newuser/PycharmProjects/DigitalImageProcessing/Output/Frequency/"

# Create a sample 400*400 image using numpy
img = np.array([[128 if (x < 100 and y < 100) else 0 for x in range(400)] for y in range(400)])
fig, axes = plot.subplots(2,2);
axes[0,0].imshow(img)
axes[0,0].title.set_text("Sample Image")

# #Find Fourier Transform (since image is 2D FT is also 2D) of image and plot
f_uv_spectrum = fourier_spectrum(img = img)
axes[0,1].imshow(f_uv_spectrum)
axes[0,1].title.set_text("Fourier Spectrum")

#Reconstrucing image from Fourier Transform
# f_uv = np.fft.fft2(img)
# reconstructed_img = np.fft.ifft2(f_uv)
# reconstructed_img = np.real(reconstructed_img)
# plot.imshow(reconstructed_img)
# plot.show()

#Currently the origin of the Fourier Spectrum's image is the origin of the Fourier spectrum
#To shift the origin of the Fourier spectrum to the center of the spectrum image (N/2, N/2),
#multiply f(x,y) (or the intensity of the original image) by (-1)^x+y
temp_img = multipy_minus_one_power_x_plus_y(img)
axes[1,0].imshow(temp_img, vmin=-255, vmax=255)
axes[1,0].title.set_text("Sample Image * -1^(x+y)")
f_uv_spectrum_origin_shifted = fourier_spectrum(img = temp_img)
axes[1,1].imshow(f_uv_spectrum_origin_shifted)
axes[1,1].title.set_text("Origin Shifted Spectrum")
plot.savefig(fname=OP_DIRECTORY+"Fourier Spectrum Illustration") #use before show() to prevent blank image being saved
plot.show()


# #Fourier of the Laplacian in Edge Detection
# #Fourier of the Laplacian = - 4*pi^2*(u^2+ v^2) * F(u,v)
ft_img = np.fft.fft2(img)
for index, f in np.ndenumerate(ft_img):
    u= index[0]
    v = index[1]
    ft_img[u,v] = -4 * pow(numpy.pi,2) * (u*u + v*v) * f

reconstructed_laplacian_img = np.fft.ifft2(ft_img)
reconstructed_laplacian_img = np.real(reconstructed_laplacian_img)
print(reconstructed_laplacian_img)
plot.imshow(reconstructed_laplacian_img)
plot.savefig(fname = OP_DIRECTORY + "Laplacian in Frequency Domain.jpeg")
plot.show()


#FREQUENCY DOMAIN FILTERING

test_img = io.imread("/Users/newuser/COURSE MATERIALS/078MSICE/SEM III/Digital Image Processing/Assignment/Images/shapes_and_a.jpg", as_gray=True)
#LOW PASS FILTERS
h_uv_ilpf = get_Ideal_LPF_FrequencyDomain(img=test_img, d0=30)
g_xy_ilpf= perform_Freq_Domain_Transform(test_img, h_uv_ilpf)
h_uv_bwlpf = get_butterworth_lpf_frequency_domain(img=test_img, d0=30, n=2)
g_xy_bwlpf= perform_Freq_Domain_Transform(test_img, h_uv_bwlpf)
h_uv_glpf = get_gaussian_lpf_frequency_domain(img=test_img, d0=30)
g_xy_glpf= perform_Freq_Domain_Transform(test_img, h_uv_glpf)
#HIGH PASS, HIGH BOOST AND HIGH FREQUENCY EMPHASIS
h_uv_ihpf = get_Ideal_HPF_FrequencyDomain(img=test_img, d0=20)
g_xy_ihpf= perform_Freq_Domain_Transform(test_img, h_uv_ihpf)
h_uv_bwhpf = get_butterworth_hpf_frequency_domain(img=test_img, d0=20, n=2)
g_xy_bwhpf= perform_Freq_Domain_Transform(test_img, h_uv_bwhpf)
h_uv_ghpf = get_gaussian_hpf_frequency_domain(img=test_img, d0=20)
g_xy_ghpf= perform_Freq_Domain_Transform(test_img, h_uv_ghpf)

# high boost
a = 1.2
h_uv_hb = (a-1) + h_uv_ghpf
g_xy_hb= perform_Freq_Domain_Transform(test_img, h_uv_hb)

# high frequency emphasis
a= 1.2
b=1.5
h_uv_hfe = a + b * h_uv_ghpf
g_xy_hfe= perform_Freq_Domain_Transform(test_img, h_uv_hfe)
# counts, bins = np.histogram(g_xy)
# plot.stairs(counts, bins)
fig, axes = plot.subplots(3,6);
axes[0,0].imshow(test_img)
axes[0,0].title.set_text("Test Image")

axes[0,1].imshow(h_uv_ilpf)
axes[0,1].title.set_text("ILPF")
axes[0,2].imshow(g_xy_ilpf)
axes[0,2].title.set_text("ILPF Img")

axes[0,3].imshow(h_uv_bwlpf)
axes[0,3].title.set_text("BWLPF")
axes[0,4].imshow(g_xy_bwlpf)
axes[0,4].title.set_text("BWLPF Img")

axes[0,5].imshow(h_uv_glpf)
axes[0,5].title.set_text("GLPF")
axes[1,0].imshow(g_xy_glpf)
axes[1,0].title.set_text("GLPF Img")

axes[1,1].imshow(h_uv_ihpf)
axes[1,1].title.set_text("IHPF")
axes[1,2].imshow(g_xy_ihpf)
axes[1,2].title.set_text("IHPF Img")

axes[1,3].imshow(h_uv_bwhpf)
axes[1,3].title.set_text("BWHPF")
axes[1,4].imshow(g_xy_bwhpf)
axes[1,4].title.set_text("BWHPF Img")

axes[1,5].imshow(h_uv_ghpf)
axes[1,5].title.set_text("GHPF")
axes[2,0].imshow(g_xy_ghpf)
axes[2,0].title.set_text("GHPF Img")

axes[2,1].imshow(h_uv_hb)
axes[2,1].title.set_text("HBF")
axes[2,2].imshow(g_xy_hb)
axes[2,2].title.set_text("HBF Img")

axes[2,3].imshow(h_uv_hfe)
axes[2,3].title.set_text("HFEF")
axes[2,4].imshow(g_xy_hfe)
axes[2,4].title.set_text("HFEF Img")

plot.axis('off')
plot.savefig(fname = OP_DIRECTORY + "Different High and Low Pass Filters in Frequency Domain.jpeg", dpi=1200)
plot.show()