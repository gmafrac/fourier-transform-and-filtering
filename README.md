# fourier-transform-and-filtering

## Introduction

This Python project implements various image filtering techniques in the frequency domain using the Discrete Fourier Transform (DFT). It leverages the NumPy and imageio libraries for efficient image processing. The `lib.py` file contains the implemented functions and the `solution.ipynb` notebook contains the processing run and analysis for the .in entries in "tests_cases_in_out".

## Functionality

The program performs the following steps:


1. Filter Implementation:

    * Implements the following filters:
    * Ideal Low-pass (index i = 0) with radius r.
    * Ideal High-pass (index i = 1) with radius r.
    * Ideal Band-stop (index i = 2) with radius r0 and r1.
    * Laplacian High-pass (index i = 3).
    * Gaussian Low-pass (index i = 4) with standard deviations σ1 and σ2.

2. Image Processing:

    * Generates the Fourier Spectrum (F(I)) of a input image I.
    * Filters F(I) by multiplying it with the chosen filter.
    * Transforms the filtered frequency domain representation back to the spatial domain using the inverse DFT, resulting in the filtered image G.
    * Calculates the Root Mean Squared Error (RMSE) between the filtered image G and the reference image H for evaluation.
    * Save the filtered image and the Spectrum in `test_cases_my_results` and `test_cases_my_results_spec`.



