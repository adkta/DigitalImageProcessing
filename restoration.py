import math
from math import floor

from skimage import io, img_as_ubyte
from spatial import zeropad_greylevel_extremities, show_image
import scipy as sp
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from frequency import distance


def get_window(img, window_size, i, j):
    return img[i:i+window_size, j:j+window_size]

def get_window_unpadded(img, window_size, i, j):
    """
    Excepts unpadded original image
    Pads 0 as necessary when returning a window
    :param img:
    :param window_size:
    :param i:
    :param j:
    :return:
    """

    if window_size < 0:
        raise Exception("Can't have a negative window size")

    if window_size % 2 != 1:
        raise Exception("Even window sizes aren't allowed")

    row_num, col_num = img.shape
    d = int((window_size - 1) / 2)

    window = np.empty((window_size, window_size))
    for k in range(window_size):
        for l in range(window_size):
            mapped_img_row = i - d + k
            mapped_img_col = j - d + l
            if mapped_img_row < 0 or mapped_img_row > row_num - 1 or mapped_img_col < 0 or mapped_img_col > col_num - 1:
                window[k, l] = 0
            else:
                window[k, l] = img[mapped_img_row, mapped_img_col]

    return window

def handle_intensity(l):
    """
    Accepts 0-255 (8 bit unsigned integer)
    Any value over 255 = 255, any value under 0 will be converted to 0, floating point will be floored
    :param l: intensity
    :return:
    """
    if l > 255:
        return 255
    if l < 0:
        return 0
    return floor(l)

def handle_intensity_ndarray(x):
    if type(x) != "ndarray":
        raise Exception("This function only accepts ndarray at this time.")

    with np.nditer(x, op_flags=['readwrite']) as it:
        for value in it:
            value[...] = handle_intensity(value)

    return x


def geometric_mean(img, window_size):
    row_num, col_num = img.shape
    img = zeropad_greylevel_extremities(img, int((window_size - 1) / 2))
    op_img = np.empty((row_num, col_num))
    for i in range(row_num - window_size + 1):
        for j in range(col_num - window_size + 1):
            op_img[i, j] = handle_intensity(sp.stats.gmean(get_window(img, window_size, i, j), axis=None))
    return op_img

def harmonic_mean(img, window_size):
    row_num, col_num = img.shape
    img = zeropad_greylevel_extremities(img, int((window_size - 1) / 2))
    op_img = np.empty((row_num, col_num))
    for i in range(row_num - window_size + 1):
        for j in range(col_num - window_size + 1):
            op_img[i, j] = handle_intensity(sp.stats.hmean(get_window(img, window_size, i, j), axis=None))
    return op_img


def contraharmonic_mean(img, window_size, q):
    row_num, col_num = img.shape
    img = zeropad_greylevel_extremities(img, int((window_size - 1) / 2))
    op_img = np.empty((row_num, col_num))
    for i in range(row_num - window_size + 1):
        for j in range(col_num - window_size + 1):
            op_img[i, j] = handle_intensity(contraharmonic_mean(get_window(img, window_size, i, j), q))
    return handle_intensity_ndarray(op_img)


def contraharmonic_mean(a, q):
    if type(a) != 'ndarray':
        print("Only accepting one numpy array as argument to this method for now")
        return None
    numerator = math.pow((a, q))
    denominator = math.pow((a,q-1))
    return np.divide(numerator, denominator)


def midpoint_filter(img, window_size):
    row_num, col_num = img.shape
    img = zeropad_greylevel_extremities(img, int((window_size - 1) / 2))
    op_img = np.empty((row_num, col_num))

    for i in range(row_num):
        for j in range(col_num):
            window = get_window(img, window_size, i, j)
            max = np.max(window)
            min = np.min(window)
            op_img[i,j] = handle_intensity((max+min)/2)

    return op_img


def alpha_trimmed_mean_filter(img, window_size, d ):
    if d%2 != 0:
        raise Exception("d must be an even number so we can trim equal number of high and low values")

    if d < 0:
        raise Exception("d cannot be negative")

    if d > window_size*window_size-1:
        raise Exception("d cannot be greater than window_size^2 - 1")

    row_num, col_num = img.shape
    img = zeropad_greylevel_extremities(img, int((window_size - 1) / 2))
    op_img = np.empty((row_num, col_num))
    for i in range(row_num):
        for j in range(col_num):
            window = get_window(img, window_size, i, j)
            sorted = np.sort(window, axis=None)
            trimmed = sorted[int(d/2):window_size-int(d/2)]
            op_img[i,j] = handle_intensity(np.mean(trimmed))
    return op_img


def get_ideal_band_pass_filter(img, d0, W):
    """
    :param img: image
    :param d0: cut-off frequency
    :param W: bandwidth
    :return:
    """
    m,n = img.shape #the spatial size will be the same as the size of the Fourier Transform of the image
    h = np.empty((m,n))
    origin_x = m/2
    origin_y = n/2
    for u in range(m):
        for v in range(n):
            d=distance(u,v,origin_x,origin_y)
            #filter logic
            if d < d0-W/2 or d > d0+W/2:
                h[u,v] = 0
            else:
                h[u,v] = 1

    return h


def get_ideal_band_reject_filter(img, d0, W):
    return 1 - get_ideal_band_pass_filter(img, d0, W)


def get_butterworth_bandpass_filter(img, d0, W, order):
    """
    :param img: image
    :param d0: cut-off frequency
    :param W: bandwidth
    :param n: order
    :return:
    """
    m,n = img.shape #the spatial size will be the same as the size of the Fourier Transform of the image
    h = np.empty((m,n))
    origin_x = m/2
    origin_y = n/2
    for u in range(m):
        for v in range(n):
            d=distance(u,v,origin_x,origin_y)
            if d == 0:
                h[u,v] = 0
            else:
                numer = pow(d,2)-pow(d0,2)
                denom = d*W
                h[u,v]=1/(1+pow(numer/denom,order))
    return h


def get_butterworth_band_reject_filter(img, d0, W, order):
    return 1 - get_butterworth_bandpass_filter(img, d0, W, order)


def get_gaussian_band_pass_filter(img, d0, W):
    """
    :param img: image
    :param d0: cut-off frequency
    :param W: bandwidth
    :param n: order
    :return:
    """
    m,n = img.shape #the spatial size will be the same as the size of the Fourier Transform of the image
    h = np.empty((m,n))
    origin_x = m/2
    origin_y = n/2
    for u in range(m):
        for v in range(n):
            d=distance(u,v,origin_x,origin_y)
            if d == 0:
                h[u,v] = 0
            else:
                numer = pow(d,2)-pow(d0,2)
                denom = d*W
                h[u,v]=pow(math.e,-1/2 * pow(numer/denom, 2))
    return h


def get_gaussian_band_reject_filter(img, d0, W):
    return 1 - get_gaussian_band_pass_filter(img, d0, W)


def adaptive_mean_filter(img, noise_var, window_size):
    row_num, col_num = img.shape
    img = zeropad_greylevel_extremities(img, int((window_size - 1) / 2))
    op_img = np.empty((row_num, col_num))
    central_pixel_del_x=(window_size-1)/2
    central_pixel_del_y=central_pixel_del_x
    for i in range(row_num):
        for j in range(col_num):
            window = get_window(img, window_size, i, j)
            local_var = np.var(window)
            noise_to_signal = noise_var/local_var
            if noise_var>local_var:
                noise_to_signal=1
            central_pixel = window[central_pixel_del_x, central_pixel_del_x]
            op_img[i,j] =  handle_intensity(central_pixel + (noise_var/local_var)*(central_pixel - np.mean(window,axis=None)))
    return op_img


def adaptive_median_filter(img, max_window_size):
    row_num, col_num = img.shape
    op_img = np.empty((row_num, col_num))
    for i in range(row_num):
        for j in range(col_num):
            z = img[i, j]
            zmed = 0
            zmax = 0
            zmin = 0
            found_req_med = False
            for window_size in range(3, max_window_size, 2):
                window = get_window_unpadded(img, window_size, i , j)
                #levelA
                zmed = np.median(window, axis=None)
                zmax = np.max(window, axis=None)
                zmin = np.min(window, axis=None)
                if zmed > zmin and zmed < zmax:
                    found_req_med = True
                    break

            if not found_req_med or not (z > zmin and z < zmax):
                op_img[i,j] = zmed
            else:
                op_img[i,j] = z
    return op_img


#Change colour map from viridis to gray
mpl.rc('image', cmap='gray')
font = {'size': 5}
mpl.rc('font', **font)

img = io.imread("/Users/newuser/COURSE MATERIALS/078MSICE/SEM III/Digital Image Processing/Assignment/Images/shapes_and_a.jpg", as_gray=True)
img = img_as_ubyte(img)
# fig, axes = plt.subplots(1,2)
# axes[0].imshow(img)
# axes[1].imshow(alpha_trimmed_mean_filter(img, window_size=3, d = 8))
# plt.show()

# show_image(get_ideal_band_pass_filter(img, 100, 20))
# show_image(get_butterworth_bandpass_filter(img, 100, 20, 2))
# show_image(get_gaussian_band_pass_filter(img, 100, 20))

op_img = adaptive_median_filter(img, 5)
print(op_img)
# show_image(op_img)

