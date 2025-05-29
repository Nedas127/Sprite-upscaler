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
from typing import Dict, Tuple
import cv2
import numpy as np


def validate_image_shapes(img1: np.ndarray, img2: np.ndarray) -> None:
    """Validate that two images have the same shape."""
    if img1.shape != img2.shape:
        raise ValueError("Images must have the same dimensions and color depth")


def get_luma_channel(image: np.ndarray) -> np.ndarray:
    """Extract the luma channel from an image (converts to YCrCb if color)."""
    if len(image.shape) == 3 and image.shape[2] > 1:
        return cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    return image


def get_max_pixel_value(image: np.ndarray) -> float:
    """Determine the maximum possible pixel value based on image dtype."""
    return np.iinfo(image.dtype).max if np.issubdtype(image.dtype, np.integer) else 1.0


def calculate_mse(original_img: np.ndarray, comparison_img: np.ndarray) -> float:
    """
    Calculate Mean Squared Error between two images.

    Args:
        original_img: Original reference image as numpy array
        comparison_img: Image to compare against the original as numpy array

    Returns:
        Mean Squared Error value (lower is better)
    """
    validate_image_shapes(original_img, comparison_img)
    error = np.subtract(original_img.astype(np.float32), comparison_img.astype(np.float32))
    return float(np.mean(np.square(error)))


def calculate_psnr(original_img: np.ndarray, comparison_img: np.ndarray,
                  max_pixel_value: float = 1.0) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio between two images.

    Args:
        original_img: Original reference image as numpy array
        comparison_img: Image to compare against the original as numpy array
        max_pixel_value: Maximum possible pixel value (default is 1.0 for normalized images)

    Returns:
        PSNR value in dB (higher is better)
    """
    mse = calculate_mse(original_img, comparison_img)
    if mse == 0:
        return float('inf')  # PSNR is infinity if images are identical
    return 10 * math.log10((max_pixel_value ** 2) / mse)


def _create_ssim_window(size: int = 11, sigma: float = 1.5) -> np.ndarray:
    """Create a Gaussian window for SSIM calculation."""
    kernel = cv2.getGaussianKernel(size, sigma)
    return np.outer(kernel, kernel.transpose())


def calculate_ssim(img1: np.ndarray, img2: np.ndarray,
                  window_size: int = 11, sigma: float = 1.5,
                  c1: float = 0.01**2, c2: float = 0.03**2) -> float:
    """
    Calculate mean localized Structural Similarity Index (SSIM) between two images.

    Args:
        img1: First image as numpy array
        img2: Second image to compare as numpy array
        window_size: Size of the Gaussian window (default: 11)
        sigma: Standard deviation for Gaussian window (default: 1.5)
        c1: Constant to stabilize division (default: 0.01**2)
        c2: Constant to stabilize division (default: 0.03**2)

    Returns:
        SSIM value between 0 and 1 (higher is better)
    """
    window = _create_ssim_window(window_size, sigma)
    padding = window_size // 2

    # Local means
    mu1 = cv2.filter2D(img1, -1, window)[padding:-padding, padding:-padding]
    mu2 = cv2.filter2D(img2, -1, window)[padding:-padding, padding:-padding]

    mu1_sq, mu2_sq, mu1_mu2 = mu1**2, mu2**2, mu1 * mu2

    # Local variances and covariance
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[padding:-padding, padding:-padding] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[padding:-padding, padding:-padding] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[padding:-padding, padding:-padding] - mu1_mu2

    # SSIM map
    numerator = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
    denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    ssim_map = numerator / denominator

    return float(np.mean(ssim_map))


def compute_metrics(original_img: np.ndarray, comparison_img: np.ndarray) -> Dict[str, float]:
    """
    Compute all image quality metrics between two images.

    If images are not grayscale, they are converted to YCrCb and metrics
    are computed on the luma (Y) channel only.

    Args:
        original_img: Original reference image as numpy array
        comparison_img: Image to compare against the original as numpy array

    Returns:
        Dictionary containing MSE, PSNR, and SSIM values
    """
    validate_image_shapes(original_img, comparison_img)

    # Use luma channel for color images
    orig_y = get_luma_channel(original_img)
    comp_y = get_luma_channel(comparison_img)

    max_val = get_max_pixel_value(original_img)

    return {
        'mse': calculate_mse(orig_y, comp_y),
        'psnr': calculate_psnr(orig_y, comp_y, max_val),
        'ssim': calculate_ssim(orig_y, comp_y)
    }


def load_and_normalize_images(original_path: str, comparison_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load and normalize two images for comparison."""
    original = cv2.imread(original_path)
    comparison = cv2.imread(comparison_path)

    if original is None or comparison is None:
        raise FileNotFoundError(
            f"Could not open images: {original_path if original is None else ''} "
            f"{comparison_path if comparison is None else ''}"
        )

    validate_image_shapes(original, comparison)
    return original.astype(np.float32) / 255.0, comparison.astype(np.float32) / 255.0


def main() -> None:
    """Command-line interface for calculating image metrics."""
    import argparse

    parser = argparse.ArgumentParser(description="Calculate image quality metrics between two images")
    parser.add_argument("original", help="Path to the original reference image")
    parser.add_argument("comparison", help="Path to the comparison image")
    args = parser.parse_args()

    try:
        original, comparison = load_and_normalize_images(args.original, args.comparison)
        metrics = compute_metrics(original, comparison)

        print(f"MSE:  {metrics['mse']:.6f}")
        print(f"PSNR: {metrics['psnr']:.6f} dB")
        print(f"SSIM: {metrics['ssim']:.6f}")
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()