from __future__ import annotations

import math
from enum import Enum, auto
from typing import Optional, Union

import numpy as np
import cv2


class EdgeFilter(Enum):
    # Existing edge detection filters
    SOBEL = "sobel"
    SOBEL_LIKE_5 = "sobel-like-5"
    SOBEL_LIKE_7 = "sobel-like-7"
    SOBEL_LIKE_9 = "sobel-like-9"
    PREWITT = "prewitt"
    SCHARR = "scharr"
    FOUR_SAMPLE = "4-sample"
    MULTI_GAUSS = "multi-gauss"


class SmoothFilter(Enum):
    # New smoothing filters
    GAUSSIAN_3x3 = auto()
    GAUSSIAN_5x5 = auto()
    BILATERAL = auto()
    EDGE_PRESERVING = auto()


# Existing edge detection kernels (unchanged)
FILTERS_X: dict[EdgeFilter, np.ndarray] = {
    EdgeFilter.SOBEL: np.array(
        [
            [+1, 0, -1],
            [+2, 0, -2],
            [+1, 0, -1],
        ]
    ),
    EdgeFilter.SOBEL_LIKE_5: np.array(
        [
            [1 / 16, 1 / 10, 0, -1 / 10, -1 / 16],
            [1 / 10, 1 / 2.8, 0, -1 / 2.8, -1 / 10],
            [1 / 8, 1 / 2.0, 0, -1 / 2.0, -1 / 8],
            [1 / 10, 1 / 2.8, 0, -1 / 2.8, -1 / 10],
            [1 / 16, 1 / 10, 0, -1 / 10, -1 / 16],
        ]
    ),
    EdgeFilter.SOBEL_LIKE_7: np.array(
        [
            [1, 2, 3, 0, -3, -2, -1],
            [2, 3, 4, 0, -4, -3, -2],
            [3, 4, 5, 0, -5, -4, -3],
            [4, 5, 6, 0, -6, -5, -4],
            [3, 4, 5, 0, -5, -4, -3],
            [2, 3, 4, 0, -4, -3, -2],
            [1, 2, 3, 0, -3, -2, -1],
        ]
    ),
    EdgeFilter.SOBEL_LIKE_9: np.array(
        [
            [1, 2, 3, 4, 0, -4, -3, -2, -1],
            [2, 3, 4, 5, 0, -5, -4, -3, -2],
            [3, 4, 5, 6, 0, -6, -5, -4, -3],
            [4, 5, 6, 7, 0, -7, -6, -5, -4],
            [5, 6, 7, 8, 0, -8, -7, -6, -5],
            [4, 5, 6, 7, 0, -7, -6, -5, -4],
            [3, 4, 5, 6, 0, -6, -5, -4, -3],
            [2, 3, 4, 5, 0, -5, -4, -3, -2],
            [1, 2, 3, 4, 0, -4, -3, -2, -1],
        ]
    ),
    EdgeFilter.PREWITT: np.array(
        [
            [+1, 0, -1],
            [+1, 0, -1],
            [+1, 0, -1],
        ]
    ),
    EdgeFilter.SCHARR: np.array(
        [
            [+3, 0, -3],
            [+10, 0, -10],
            [+3, 0, -3],
        ]
    ),
    EdgeFilter.FOUR_SAMPLE: np.array(
        [
            [1, 0, -1],
        ]
    ),
}

# New smoothing kernels
SMOOTHING_FILTERS = {
    SmoothFilter.GAUSSIAN_3x3: np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ]) / 16,

    SmoothFilter.GAUSSIAN_5x5: np.array([
        [1, 4, 6, 4, 1],
        [4, 16, 24, 16, 4],
        [6, 24, 36, 24, 6],
        [4, 16, 24, 16, 4],
        [1, 4, 6, 4, 1]
    ]) / 256,
}


def create_gauss_kernel(parameters: list[tuple[float, float]], for_smoothing: bool = False) -> np.ndarray:
    """Modified to support both edge detection and smoothing"""
    total_volume = sum(weight for _, weight in parameters)
    if total_volume == 0:
        return np.zeros((1, 1))

    def sample(x: float, y: float) -> float:
        s = 0
        for o, weight in parameters:
            std2 = 2 * o * o
            s += weight / (math.pi * std2) * np.exp(-(x * x + y * y) / std2)
        return s

    kernel_radius = 1
    for o, weight in parameters:
        if weight > 0:
            kernel_radius = max(kernel_radius, math.ceil(2 * o))
    kernel_radius += 1

    kernel_size = 2 * kernel_radius + 1
    kernel = np.zeros((kernel_size, kernel_size))
    x_offsets = [0, 0.25, 0.5, 0.75]

    for y in range(kernel_size):
        y_pos = y - kernel_radius
        for x in range(kernel_size):
            x_pos = x - kernel_radius
            s = 0
            for x_offset in x_offsets:
                s += sample(abs(x_pos) - 1 + x_offset, y_pos)

            if for_smoothing:
                kernel[y, x] = s / len(x_offsets)
            else:
                kernel[y, x] = s / len(x_offsets) * -np.sign(x_pos)

    if for_smoothing:
        kernel = kernel / np.sum(kernel)  # Normalize for smoothing
    elif kernel_size > 3:  # Normalize edge detection kernels
        left = kernel[:, :kernel_size // 2]
        kernel = kernel / np.sum(left)

    return kernel


def get_filter_kernels(
        edge_filter: EdgeFilter,
        gauss_parameters: list[tuple[float, float]] = None,
        for_smoothing: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Modified to support both edge detection and smoothing"""
    if edge_filter == EdgeFilter.MULTI_GAUSS:
        if gauss_parameters is None:
            gauss_parameters = [(1.0, 1.0)]
        filter_x = create_gauss_kernel(gauss_parameters, for_smoothing)
    else:
        filter_x = FILTERS_X.get(edge_filter)
        if filter_x is None:
            raise ValueError(f"Unknown filter '{edge_filter}'")

        if not for_smoothing and filter_x.shape[0] > 3:  # Normalize larger edge detection kernels
            left = filter_x[:, :filter_x.shape[1] // 2]
            filter_x = filter_x / np.sum(left)

    filter_y = np.rot90(filter_x, -1)
    return filter_x, filter_y


def smooth_image(
        image: np.ndarray,
        method: Union[EdgeFilter, SmoothFilter] = SmoothFilter.GAUSSIAN_3x3,
        gauss_parameters: Optional[list[tuple[float, float]]] = None,
        edge_mask: Optional[np.ndarray] = None,
        **kwargs
) -> np.ndarray:
    """
    Main smoothing function that works with both edge-preserving and regular smoothing.

    Args:
        image: Input image (H,W) or (H,W,C)
        method: Either an EdgeFilter or SmoothFilter enum value
        gauss_parameters: Parameters for MULTI_GAUSS filter
        edge_mask: Optional mask where smoothing should be applied
        kwargs: Additional parameters for specific methods (diameter, sigmaColor, sigmaSpace for bilateral)

    Returns:
        Smoothed image
    """
    if isinstance(method, SmoothFilter):
        if method == SmoothFilter.BILATERAL:
            return cv2.bilateralFilter(
                image,
                d=kwargs.get('diameter', 9),
                sigmaColor=kwargs.get('sigmaColor', 75),
                sigmaSpace=kwargs.get('sigmaSpace', 75)
            )
        elif method == SmoothFilter.EDGE_PRESERVING:
            temp = cv2.edgePreservingFilter(image, flags=cv2.RECURS_FILTER)
            return cv2.addWeighted(image, 0.5, temp, 0.5, 0)
        else:
            kernel = SMOOTHING_FILTERS[method]
    else:  # EdgeFilter
        kernel_x, _ = get_filter_kernels(method, gauss_parameters, for_smoothing=True)
        kernel = kernel_x

    if edge_mask is None:
        return cv2.filter2D(image, -1, kernel)
    else:
        smoothed = cv2.filter2D(image, -1, kernel)
        if len(image.shape) == 3 and len(edge_mask.shape) == 2:
            edge_mask = edge_mask[..., None]
        return np.where(edge_mask > 0, smoothed, image)


def detect_edges(
        image: np.ndarray,
        method: EdgeFilter = EdgeFilter.SOBEL,
        gauss_parameters: Optional[list[tuple[float, float]]] = None,
        threshold: float = 0.1
) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
    kernel_x, kernel_y = get_filter_kernels(method, gauss_parameters)

    # Convert to float32 before filtering
    gray = gray.astype(np.float32)

    grad_x = cv2.filter2D(gray, -1, kernel_x)
    grad_y = cv2.filter2D(gray, -1, kernel_y)

    # Ensure both gradients are float32 type
    grad_x = grad_x.astype(np.float32)
    grad_y = grad_y.astype(np.float32)

    magnitude = cv2.magnitude(grad_x, grad_y)
    return (magnitude > (threshold * 255)).astype(np.uint8) * 255


def smooth_edges(
        image: np.ndarray,
        edge_detection: EdgeFilter = EdgeFilter.SOBEL,
        smoothing: Union[EdgeFilter, SmoothFilter] = SmoothFilter.GAUSSIAN_3x3,
        edge_threshold: float = 0.1,
        **kwargs
) -> np.ndarray:
    """
    Complete edge smoothing pipeline:
    1. Detect edges
    2. Apply selective smoothing
    """
    edges = detect_edges(image, edge_detection, threshold=edge_threshold)
    return smooth_image(image, smoothing, edge_mask=edges, **kwargs)


# Example usage
if __name__ == "__main__":
    # Load an image
    image = cv2.imread("input.png")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Option 1: Simple smoothing
    smoothed = smooth_image(image, SmoothFilter.GAUSSIAN_5x5)

    # Option 2: Edge-preserving smoothing
    edge_preserved = smooth_image(image, SmoothFilter.BILATERAL, diameter=9, sigmaColor=75, sigmaSpace=75)

    # Option 3: Selective edge smoothing
    edge_smoothed = smooth_edges(
        image,
        edge_detection=EdgeFilter.SCHARR,
        smoothing=SmoothFilter.GAUSSIAN_3x3,
        edge_threshold=0.08
    )

    # Save results
    cv2.imwrite("smoothed.png", cv2.cvtColor(smoothed, cv2.COLOR_RGB2BGR))
    cv2.imwrite("edge_preserved.png", cv2.cvtColor(edge_preserved, cv2.COLOR_RGB2BGR))
    cv2.imwrite("edge_smoothed.png", cv2.cvtColor(edge_smoothed, cv2.COLOR_RGB2BGR))