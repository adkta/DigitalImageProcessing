import numpy as np
from matplotlib import pyplot as plt
from skimage import img_as_ubyte


def show_image(img):
    plt.imshow(img)
    plt.show()


def bit_plane_slicing(bit_plane, img):
    """
    Use bit wise operator np.bitwise_and to perform AND operator and thus slice the image based on bit_plane
    :param bit_plane:
    :param img: unsigned integer
    :return: sliced image
    """

    allowed_img_dtypes = ('uint8', 'float64')
    if img.dtype not in allowed_img_dtypes:
        print("Image neither in 8 bit gray scale nor normalized gray scale! No operation will be performed. Returning...")
        return

    temp_img = img
    if img.dtype == 'float64':
        temp_img == img_as_ubyte(img)

    slice_mask = pow(2, bit_plane)
    return np.bitwise_and(slice_mask,temp_img)


def negative(img):
    """
    Current input can only be 8 bit unsigned integer (0-255).
    More specifically dtype = uint8
    :param img: Unsigned 8 bit integer
    :return:
    """
    return np.subtract(255, img)


def zeropad_greylevel_extremities(img, pad_num):
    temp = img.copy()

    for i in range(pad_num):
        nr, nc = temp.shape
        #insert along rows (axis 0)
        temp = np.insert(temp, nr, 0, axis=0)
        temp = np.insert(temp, 0, 0, axis=0)
        #insert along columns (axis 1)
        temp = np.insert(temp, nc, 0, axis=1)
        temp = np.insert(temp, 0, 0, axis=1)

    return temp


def apply_mask(img, mask, absolute = False, handle_range = True):
    """
    Mask is applied by zero padding the missing pixels (i.e. pixels at extremities).
    This will allow o/p image to be the same size as i/p image
    This means we surround the image by 0s
    Be careful that this will smooth out the boundaries of the image
    Additionally, mixing pixels might be addressed through one of:
        1)Wrapping around the other boundary
        2)Duplicating boundary row or columns
        3)Skipping boundary rows
        4)Neglecting missing pixels (i.e. applying the elements of mask only partially if missing). Similar to padding.
    But we are only addressing the mixing pixel scenario
    :param img:
    :param mask:
    :return:
    """

    if type(mask) != 'ndarray':
        mask = np.array(mask)

    m_rownum, m_colnum = mask.shape

    #Do not compute if the mask isn't square matrix
    if m_rownum != m_colnum:
        print("Invalid mask. Row num and column num do not match.")
        return None

    pad_num = int((m_rownum - 1)/2)   #Calculate no. of zero paddings or zero surroundings required based on mask size
    zeropadded_image = zeropad_greylevel_extremities(img, pad_num)

    img_rownum, img_colnum = zeropadded_image.shape

    op_img = np.empty(shape=(img_rownum-pad_num*2, img_colnum-pad_num*2))

    for i in range(img_rownum - pad_num*2): #move across rows
        for j in range(img_colnum - pad_num*2): #move accross each column in rows
            window = zeropadded_image[i:i+m_rownum, j:j+m_colnum]
            pixel_intensity = np.sum(mask * window, axis=None) #apply mask to each window. axis=None will sum all elements of array, not just along an axis

            if absolute:
                pixel_intensity = abs(pixel_intensity)

            if handle_range:
                pixel_intensity = handle_out_of_range(pixel_intensity)  #handle out of range cases

            op_img[i,j] = pixel_intensity

    return op_img


def handle_out_of_range(pixel_intensity):
    if pixel_intensity > 1.0:
        return 1.0
    elif pixel_intensity < 0.0:
        return 0.0
    else:
        return pixel_intensity


def averaging_filter(img, window_size):
    avg_filter_mask = 1 / pow(window_size, 2) * np.array([[1.0 for i in range(window_size)] for j in range(window_size)])
    return apply_mask(img, avg_filter_mask)


def nonlinear_filter(img, window_size, type):
    if window_size%2 == 0:
        print(f"Window size should be odd. Provided: {window_size}")
        return None

    allowed_types = ("max", "min", "median")
    if type not in allowed_types:
        print(f"Allowed types of filters are {allowed_types}. Provided: {type}")
        return None

    nr,nc = img.shape;
    # Computing median requires sorting and thus comparison.
    # Comparison operation is less expensive when converted to uint8 instead of float64
    temp_img = img_as_ubyte(img)
    pad_num = int((window_size - 1)/2)
    temp_img = zeropad_greylevel_extremities(img, pad_num)
    op_img = np.empty(shape=(nr,nc))

    if type == "median":
        get_median(nc, nr, op_img, temp_img, window_size)
    elif type == "max":
        get_max(nc, nr, op_img, temp_img, window_size)
    else:
        get_min(nc, nr, op_img, temp_img, window_size)

    return op_img


def get_median(nc, nr, op_img, temp_img, window_size):
    for i in range(nr):
        for j in range(nc):
            op_img[i, j] = np.median(temp_img[i:i + window_size, j:j + window_size])


def get_max(nc, nr, op_img, temp_img, window_size):
    for i in range(nr):
        for j in range(nc):
            op_img[i, j] = np.max(temp_img[i:i + window_size, j:j + window_size])


def get_min(nc, nr, op_img, temp_img, window_size):
    for i in range(nr):
        for j in range(nc):
            op_img[i, j] = np.min(temp_img[i:i + window_size, j:j + window_size])


def high_pass_filter(img, mask =1 / 9 * np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])):
    return apply_mask(img,mask)


def high_filter(img, a, b = 1, filter_type = "High Boost"):
    # HBF = (A-1)*Original + HPF, A > 1
    # HFEF = A*Original + HPF , A > 1, B > A
    allowable_filter_types = ["High Boost", "High Frequency Emphasis"]

    if filter_type not in allowable_filter_types:
        print(f"type should be one of the following allowable filter types: {allowable_filter_types}. Provided: {filter_type}")
        return None

    if a < 1:
        print(f"Value of a should not be less than 1. Provided: {a}")
        return None

    if filter_type == "High Frequency Emphasis" and b <= a:
        print(f"b > a in case of HFE filters. Provided: {b}")
        return None

    if filter_type == "High Boost":
        a = a-1

    op_img = a * img + b * high_pass_filter(img=img)
    handle_out_of_range_ndarray(op_img)
    return op_img


def handle_out_of_range_ndarray(img):
    with np.nditer(img, op_flags=['readwrite']) as it:
        for x in it:#specifying op_flags as write also since we wan't to modify the image intensity ndarray
            x[...]=handle_out_of_range(x)


def laplacian(img, mask = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])):
    """
    Actually the laplacian is [[0,-1,0],[-1,4,-1],[0,-1,0]].
    The default does f(x,y) + laplacian (similar to highboost without the A-1 term).
    The mask parameter can be slight variants of laplacian such as [[0,-1,0],[-1,8,-1],[0,-1,0]], [[0,-1,0],[-1,9,-1],[0,-1,0]], etc
    :param img:
    :param mask:
    :return:
    """
    return apply_mask(img,mask)


def sobel(img):
    sobel_operator_x = np.array([[-1,-2,-1],[0, 0, 0],[1, 2, 1]])
    sobel_operator_y = np.transpose(sobel_operator_x)
    sobel_op = first_derivative_filter(img, sobel_operator_x, sobel_operator_y)
    return sobel_op


def prewitt(img):
    prewitt_operator_x = np.array([[-1,-1,-1],[0,0,0],[1,1,1]])
    prewitt_operator_y = np.transpose(prewitt_operator_x)
    prewitt_op = first_derivative_filter(img, prewitt_operator_x, prewitt_operator_y)
    return prewitt_op


def first_derivative_filter(img, operator_x, operator_y):
    x = apply_mask(img=img, mask=operator_x, absolute=True, handle_range=False)
    y = apply_mask(img=img, mask=operator_y, absolute=True, handle_range=False)
    op = x + y
    handle_out_of_range_ndarray(op)
    return op
