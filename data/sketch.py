import matlab.engine
import numpy as np
from scipy.ndimage import sobel, generic_gradient_magnitude

from image_utils import normalize_image


def generate_sketch(img, low_val=0.1, high_val=0.2):
    """Generate an image sketch, using a Canny filter with a lower threshold and a high threshold. This implementation
    uses the MATLAB implementation and thus a MATLAB engine, since no 3D python implementation is available. If you dont
    have MATLAB simply use the gradient image or a sobel filter (method available in numpy and scipy)
     :param img: input image
     :param low_val: lower threshold of Canny filter
     :param high_val: higher threshold of Canny filter
     :return: sketch of image (canny edges weighted by gradient magnitude)
     """
    norm_img = normalize_image(img, 0, 1, 'float32')

    # start MATLAB engine
    eng = matlab.engine.start_matlab()
    img_list = matlab.double(norm_img.tolist())

    # apply canny
    edges = eng.edge3(img_list, 'approxcanny', matlab.double([low_val, high_val]))
    # form MATLAB to numpy
    edges_np = edges._data
    edges_np = np.reshape(edges_np, (img.shape[2], img.shape[1], img.shape[0]))
    edges_np = np.transpose(edges_np, (2, 1, 0))
    # we want a magnitude weighted edge  image
    magnitudes = generic_gradient_magnitude(normalize_image(norm_img, 0, 1, 'float32'), sobel)
    norm_magnitudes = normalize_image(magnitudes, 0, 1, 'float32')
    return edges_np * norm_magnitudes * 255
