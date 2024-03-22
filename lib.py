# Name: Guilherme Mafra da Costa
# USP Number: 11272015
# Course Code: SCC0251
# Year/Semester: 2024/01
# Title: Fourier Transform and Filtering

from typing import Callable
import numpy as np

import matplotlib.pyplot as plt
import imageio.v3 as iio


def ideal_low_pass(shape, radius: int):
    """
    Creates an ideal low-pass filter mask.

    Args:
        shape (tuple): Shape of the mask.
        radius (int): Radius of the circular region to pass.

    Returns:
        np.array: The ideal low-pass filter mask.
    """
    mask = np.zeros(shape, dtype=np.complex128)
        
    P, Q = mask.shape
    u, v = np.indices((P, Q))

    condition = np.sqrt((u - P//2)**2 + (v - Q//2)**2) <= radius
    mask[condition] = 1

    return mask

def ideal_high_pass(shape, radius: int):
    """
    Creates an ideal high-pass filter mask.

    Args:
        shape (tuple): Shape of the mask.
        radius (int): Radius of the circular region to block.

    Returns:
        np.array: The ideal high-pass filter mask.
    """
    mask = np.ones(shape, dtype=np.complex128)
        
    P, Q = mask.shape
    u, v = np.indices((P, Q))

    condition = np.sqrt((u - P//2)**2 + (v - Q//2)**2) <= radius
    mask[condition] = 0

    return mask

def ideal_band_stop(shape, radius_min: int, radius_max: int):
    """
    Creates an ideal band-stop filter mask.

    Args:
        shape (tuple): Shape of the mask.
        radius_min (int): Minimum radius of the circular region to block.
        radius_max (int): Maximum radius of the circular region to block.

    Returns:
        np.array: The ideal band-stop filter mask.
    """
    mask = np.ones(shape, dtype=np.complex128)
        
    P, Q = mask.shape
    u, v = np.indices((P, Q))

    condition1 = np.sqrt((u - P//2)**2 + (v - Q//2)**2) >= radius_min
    condition2 = np.sqrt((u - P//2)**2 + (v - Q//2)**2) <= radius_max
    
    mask[condition1 & condition2] = 0

    return mask

def laplacian_high_pass(shape):
    """
    Creates a Laplacian high-pass filter mask.

    Args:
        shape (tuple): Shape of the mask.

    Returns:
        np.array: The Laplacian high-pass filter mask.
    """
    mask = np.zeros(shape, dtype=np.complex128)
    
    P, Q = mask.shape
    u, v = np.indices((P, Q))
    mask[u, v] = -4 * np.pi**2 * ((u - P/2)**2 + (v - Q/2)**2) / (P*Q)
    
    return mask

def gaussian_low_pass(shape, std_dev_r: float, std_dev_c: float):
    """
    Creates a Gaussian low-pass filter mask.

    Args:
        shape (tuple): Shape of the mask.
        std_dev_r (float): Standard deviation in the row direction.
        std_dev_c (float): Standard deviation in the column direction.

    Returns:
        np.array: The Gaussian low-pass filter mask.
    """
    mask = np.zeros(shape, dtype=np.complex128)
    
    P, Q = mask.shape
    u, v = np.indices((P, Q))
    
    x = ((u - P//2)**2 / (2 * std_dev_r**2)) + ((v - Q//2)**2 / (2 * std_dev_c**2))
    mask[u, v] = np.exp(-x)
    
    return mask

def process_image(img: np.array, filter: Callable[[np.array], np.array], **kwargs):
    """
    Applies a given filter to an image.

    Args:
        img (np.array): Input image.
        filter (Callable[[np.array], np.array]): Filter function to apply.
        **kwargs: Additional keyword arguments for the filter function.

    Returns:
        np.array: Processed image.
    """
    mask = filter(img.shape, **kwargs) 
    
    fourier_spectrum = np.fft.fftshift(np.fft.fft2(img, axes=(0,1)))
    spectrum_filtered = fourier_spectrum*mask   
    output_img = np.fft.ifft2(np.fft.ifftshift(spectrum_filtered), axes=(0,1)).real
    
    output_img -= output_img.min()
    output_img = (output_img * 255 / output_img.max()).astype(np.uint8)

    return output_img, spectrum_filtered

def rmse(img_high: np.array, img_high_calculated: np.array, print_error: bool = True):
    """
    Calculates the Root Mean Squared Error (RMSE) between two images.

    Args:
        img_high (np.array): High-resolution reference image.
        img_high_calculated (np.array): Calculated high-resolution image.
        print_error (bool, optional): Whether to print the error. Defaults to True.

    Returns:
        float: The RMSE value.
    """
    error = np.sqrt(((img_high - img_high_calculated)**2).sum() / img_high.size)
    
    if print_error is True:
        print(f"{error:.4f}") 
    return error

def get_input(img_in_file: str, reference_img_file: str, test: bool = False):
    """
    Retrieves input images for processing.

    Args:
        img_in_file (str): File name of the input image.
        reference_img_file (str): File name of the reference image.
        test (bool, optional): Whether it's a test case. Defaults to True.

    Returns:
        tuple: Input image and reference image.
    """
    if test:
        TEST_PATH = "test_cases_data/"
        REFERENCE_PATH = "test_cases_reference/"
    else:
        TEST_PATH = ""
        REFERENCE_PATH = ""
        
    img = iio.imread(f"{TEST_PATH}{img_in_file}")
    img_ref = iio.imread(f"{REFERENCE_PATH}{reference_img_file}")
    
    return img, img_ref