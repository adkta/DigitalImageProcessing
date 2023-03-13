from math import e

import numpy as np


def fourier_spectrum( img ):
    f_uv = np.fft.fft2(img)
    f_uv_spectrum = np.abs(f_uv)
    # Since FT absolute value (or magnitude) is usually out of range. We perform:
    # c*log(1+|F(u,v)|)/log(1+max(F(u,v))).
    # If c = 255, this will be normalized to 256 point gray level i.e. 8 bit integer
    f_uv_spectra_log_max = np.log10(1 + np.max(f_uv_spectrum))
    f_uv_spectrum = np.floor(255 * (np.log10(1 + f_uv_spectrum)) / f_uv_spectra_log_max)
    return f_uv_spectrum


def multipy_minus_one_power_x_plus_y(img):
    temp_img = np.empty(shape=img.shape)
    for index, f in np.ndenumerate(img):
        x = index[0]
        y = index[1]
        temp_img[x, y] = f * pow(-1, x + y)
    return temp_img


def perform_Freq_Domain_Transform(img, h_uv):
    #Making sure that the FT of the image has origin at the center and not top left corner
    temp_img = multipy_minus_one_power_x_plus_y(img=img)
    f_uv = np.fft.fft2(temp_img)
    g_uv = f_uv * h_uv
    g_xy = np.real(np.fft.ifft2(g_uv))
    g_xy = multipy_minus_one_power_x_plus_y(img = g_xy)
    return g_xy


def distance(u, v, img_center_x, img_center_y):
    return pow(pow(u-img_center_x,2)+pow(v-img_center_y,2),1/2)


def get_Ideal_LPF_FrequencyDomain(img, d0):
    #ft of the image is going to be the same shape as the image.
    #since the filter is going to be multiplied with FT (element by element ... also called broadcast operation in numpy
    #the filter needs to be the same shape as the image
    M,N = img.shape
    return np.array(
        [[1 if distance(u, v, M/2, N/2) <= d0 else 0 for v in range(N)] for u in range(M)])


def get_Ideal_HPF_FrequencyDomain(img, d0):
    #ft of the image is going to be the same shape as the image.
    #since the filter is going to be multiplied with FT (element by element ... also called broadcast operation in numpy
    #the filter needs to be the same shape as the image
    M,N = img.shape
    return np.array(
        [[1 if distance(u, v, M/2, N/2) > d0 else 0 for v in range(N)] for u in range(M)])


def get_butterworth_lpf_frequency_domain(img, d0, n):
    """

    :param img:
    :param d0: cutoff frequency
    :param n: order of the filter
    :return: filter (3d representation of the filter in a 2d array where the element values are 3d)
    """
    #ft of the image is going to be the same shape as the image.
    #since the filter is going to be multiplied with FT (element by element ... also called broadcast operation in numpy
    #the filter needs to be the same shape as the image
    M,N = img.shape
    return np.array([[1 / (1 + pow((distance(u, v, M/2, N/2) / d0), 2*n)) for v in range(N)] for u in range(M)])


def get_butterworth_hpf_frequency_domain(img, d0, n):
    """

    :param img:
    :param d0: cutoff frequency
    :param n: order of the filter
    :return: filter (3d representation of the filter in a 2d array where the element values are 3d)
    """
    #ft of the image is going to be the same shape as the image.
    #since the filter is going to be multiplied with FT (element by element ... also called broadcast operation in numpy
    #the filter needs to be the same shape as the image
    M,N = img.shape
    return np.array([[
        0 if u==M/2 and v==N/2 else 1 / (1 + pow((d0/distance(u, v, M/2, N/2)), 2*n))
    for v in range(N)] for u in range(M)])


def get_gaussian_lpf_frequency_domain(img, d0):
    """

    :param img:
    :param d0: cutoff frequency (or radius)
    :param n: order of the filter
    :return:
    """
    m,n = img.shape
    return np.array([[pow(e, (-1/2)*pow(distance(u,v,m/2,n/2)/d0, 2)) for v in range(n)] for u in range(m)])


def get_gaussian_hpf_frequency_domain(img, d0):
    """

    :param img:
    :param d0: cutoff frequency (or radius)
    :param n: order of the filter
    :return:
    """
    m,n = img.shape
    return np.array([[1-pow(e, (-1/2)*pow(distance(u,v,m/2,n/2)/d0, 2)) for v in range(n)] for u in range(m)])
