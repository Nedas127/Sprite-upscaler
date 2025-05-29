"""
ESRGAN Package
=============
Image Upscaling Pipeline with Transparency Handling

This package provides a framework for upscaling images with multiple AI models
while preserving transparency through dual background technique.
"""

__version__ = "1.0.0"
__author__ = "Nedas Kontenis"

# Import main classes for easy access
from .config.config_models import PipelineConfig, ResizeConfig
from .core.pipeline import UpscalingPipeline
from .core.model_processor import ModelProcessor
from .core.metrics_calculator import MetricsCalculator
from .utils.image_utils import ImageUtils
from .utils.arg_parser import parse_args

__all__ = [
    'PipelineConfig',
    'ResizeConfig',
    'UpscalingPipeline',
    'ModelProcessor',
    'MetricsCalculator',
    'ImageUtils',
    'parse_args'
]
