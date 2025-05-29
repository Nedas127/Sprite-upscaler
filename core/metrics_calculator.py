"""
Metrics Calculator
-----------------
Handles image quality metrics calculations.
"""

import cv2
import numpy as np
from typing import Dict

from utils.metrics import compute_metrics

# Type alias for clarity
ImageArray = np.ndarray


class MetricsCalculator:
    """Handles image quality metrics calculations"""

    @staticmethod
    def calculate_metrics(original_img: ImageArray, upscaled_img: ImageArray) -> Dict[str, float]:
        """
        Calculate image quality metrics between original and upscaled images.

        Args:
            original_img: Original image as numpy array
            upscaled_img: Upscaled image as numpy array

        Returns:
            Dictionary of metrics (mse, psnr, ssim)
        """
        # Convert images to RGB if they have an alpha channel
        original_rgb = original_img[:, :, :3] if original_img.shape[2] == 4 else original_img
        upscaled_rgb = upscaled_img[:, :, :3] if upscaled_img.shape[2] == 4 else upscaled_img

        # Resize original to match upscaled dimensions for fair comparison
        original_resized = cv2.resize(
            original_rgb,
            (upscaled_rgb.shape[1], upscaled_rgb.shape[0]),
            interpolation=cv2.INTER_LANCZOS4
        )

        # Normalize to [0,1] range for metric calculations
        original_norm = original_resized.astype(np.float32) / 255.0
        upscaled_norm = upscaled_rgb.astype(np.float32) / 255.0

        # Calculate metrics
        metrics = compute_metrics(original_norm, upscaled_norm)

        # Handle case where MSE is zero (PSNR would be infinity)
        if metrics['mse'] == 0:
            metrics['psnr'] = 100.0  # 100 dB is extremely good quality

        return metrics
