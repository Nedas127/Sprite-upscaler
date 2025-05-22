"""
Image Metrics Module

This module provides functions to calculate common image quality metrics:
- Mean Squared Error (MSE)
- Peak Signal-to-Noise Ratio (PSNR)
- Structural Similarity Index (SSIM)

These metrics can be used to compare an original image with a processed/compressed version
to assess the quality degradation.
"""

import math
import cv2
import numpy as np


def calculate_mse(original_img: np.ndarray, comparison_img: np.ndarray) -> float:
    """
    Calculate Mean Squared Error between two images.

    Args:
        original_img: Original reference image as numpy array
        comparison_img: Image to compare against the original as numpy array

    Returns:
        float: Mean Squared Error value (lower is better)
    """
    if original_img.shape != comparison_img.shape:
        raise ValueError("Images must have the same dimensions and color depth")

    mse = np.mean((original_img.astype(np.float32) - comparison_img.astype(np.float32)) ** 2)
    return float(mse)


def calculate_psnr(original_img: np.ndarray, comparison_img: np.ndarray, max_pixel_value: float = 1.0) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio between two images.

    Args:
        original_img: Original reference image as numpy array
        comparison_img: Image to compare against the original as numpy array
        max_pixel_value: Maximum possible pixel value (default is 1.0 for normalized images)

    Returns:
        float: PSNR value in dB (higher is better)
    """
    mse = calculate_mse(original_img, comparison_img)
    if mse == 0:
        return float('inf')  # PSNR is infinity if images are identical

    psnr = 10 * math.log10((max_pixel_value ** 2) / mse)
    return float(psnr)


def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Calculate mean localized Structural Similarity Index (SSIM) between two images.

    Args:
        img1: First image as numpy array
        img2: Second image to compare as numpy array

    Returns:
        float: SSIM value between 0 and 1 (higher is better)
    """
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]

    mu1_sq = np.power(mu1, 2)
    mu2_sq = np.power(mu2, 2)
    mu1_mu2 = np.multiply(mu1, mu2)

    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
            (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )

    return float(np.mean(ssim_map))


def compute_metrics(original_img: np.ndarray, comparison_img: np.ndarray) -> dict:
    """
    Compute all image quality metrics between two images.

    If images are not grayscale, they are converted to YCrCb and metrics
    are computed on the luma (Y) channel only.

    Args:
        original_img: Original reference image as numpy array
        comparison_img: Image to compare against the original as numpy array

    Returns:
        dict: Dictionary containing MSE, PSNR, and SSIM values
    """
    if original_img.shape != comparison_img.shape:
        raise ValueError("Images must have the same dimensions and color depth")

    # If images are not grayscale, convert to YCrCb and use luma channel
    if len(original_img.shape) == 3 and original_img.shape[2] > 1:
        orig_y = cv2.cvtColor(original_img, cv2.COLOR_BGR2YCrCb)[:, :, 0]
        comp_y = cv2.cvtColor(comparison_img, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    else:
        orig_y = original_img
        comp_y = comparison_img

    # Calculate metrics
    mse = calculate_mse(orig_y, comp_y)

    # For PSNR, determine max pixel value based on data type
    if np.issubdtype(original_img.dtype, np.integer):
        max_val = np.iinfo(original_img.dtype).max
    else:
        max_val = 1.0  # Assuming normalized floating point

    psnr = calculate_psnr(orig_y, comp_y, max_val)
    ssim = calculate_ssim(orig_y, comp_y)

    return {
        'mse': float(mse),
        'psnr': float(psnr),
        'ssim': float(ssim)
    }


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Calculate image quality metrics between two images")
    parser.add_argument("original", help="Path to the original reference image")
    parser.add_argument("comparison", help="Path to the comparison image")
    args = parser.parse_args()

    # Load images
    original = cv2.imread(args.original)
    comparison = cv2.imread(args.comparison)

    if original is None:
        raise FileNotFoundError(f"Could not open original image file: {args.original}")
    if comparison is None:
        raise FileNotFoundError(f"Could not open comparison image file: {args.comparison}")

    # Ensure images have the same dimensions
    if original.shape != comparison.shape:
        raise ValueError(f"Images have different dimensions: {original.shape} vs {comparison.shape}")

    # Normalize to [0, 1] range for floating point calculations
    original = original.astype(np.float32) / 255.0
    comparison = comparison.astype(np.float32) / 255.0

    # Calculate metrics
    metrics = compute_metrics(original, comparison)

    # Print results
    print(f"MSE:  {metrics['mse']:.6f}")
    print(f"PSNR: {metrics['psnr']:.6f} dB")
    print(f"SSIM: {metrics['ssim']:.6f}")