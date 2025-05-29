"""
Image Utility Functions
-----------------------
This module contains utility functions for image processing operations.
"""

import cv2
import numpy as np
from PIL import Image
from typing import Tuple

RGBArray = np.ndarray  # RGB image as numpy array
RGBAArray = np.ndarray  # RGBA image as numpy array


class ImageUtils:
    """Utility class for image processing operations"""

    @staticmethod
    def resize_image(image: RGBAArray, dimensions: Tuple[int, int] = None, scale: float = 1.0) -> RGBAArray:
        """
        Resize image with high quality interpolation.

        Args:
            image: RGBA numpy array
            dimensions: Target dimensions (width, height)
            scale: Scaling factor

        Returns:
            Resized RGBA numpy array
        """
        pil_img = Image.fromarray(image)

        if dimensions:
            # Resize to specific dimensions
            resized = pil_img.resize(dimensions, Image.LANCZOS)
        else:
            # Resize by scale factor
            new_width = int(image.shape[1] * scale)
            new_height = int(image.shape[0] * scale)
            resized = pil_img.resize((new_width, new_height), Image.LANCZOS)

        return np.array(resized)

    @staticmethod
    def extract_transparency(white_bg: RGBArray, black_bg: RGBArray) -> Tuple[RGBArray, np.ndarray]:
        """
        Create alpha channel from differences between white/black backgrounds.

        Args:
            white_bg: RGB image upscaled on white background
            black_bg: RGB image upscaled on black background

        Returns:
            Tuple of (RGB array, alpha channel array)
        """
        white = white_bg.astype(np.float32)
        black = black_bg.astype(np.float32)

        with np.errstate(divide='ignore', invalid='ignore'):
            diff = white - black
            alpha = 1.0 - np.clip(np.mean(diff, axis=2) / 255.0, 0, 1)
            alpha = (alpha * 255).clip(0, 255).astype(np.uint8)

            alpha_float = np.where(alpha > 0, alpha / 255.0, 1.0)
            rgb = (black / alpha_float[..., None]).clip(0, 255).astype(np.uint8)

        return rgb, alpha

    @staticmethod
    def post_process_alpha(alpha: np.ndarray) -> np.ndarray:
        """
        Clean up alpha channel with edge refinement.

        Args:
            alpha: Raw alpha channel array

        Returns:
            Refined alpha channel array
        """
        alpha = cv2.medianBlur(alpha, 3)
        edges = cv2.Canny(alpha, 50, 150)
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

        try:
            if hasattr(cv2, 'ximgproc'):
                guided_filter = cv2.ximgproc.createGuidedFilter(alpha, radius=5, eps=0.01)
                alpha = guided_filter.filter(alpha)
            else:
                alpha = cv2.bilateralFilter(alpha, 5, 75, 75)
        except Exception:
            alpha = cv2.bilateralFilter(alpha, 5, 75, 75)

        return alpha

    @staticmethod
    def prepare_background(img_pil: Image.Image, bg_color: Tuple[int, int, int]) -> Image.Image:
        """
        Create image with specified background color.

        Args:
            img_pil: PIL Image to process
            bg_color: Background color as RGB tuple

        Returns:
            Image with background applied
        """
        bg = Image.new('RGB', img_pil.size, bg_color)
        if img_pil.mode == 'RGBA':
            bg.paste(img_pil, (0, 0), img_pil)
        else:
            bg.paste(img_pil, (0, 0))
        return bg
