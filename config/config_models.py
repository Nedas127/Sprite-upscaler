"""
Configuration Models
-------------------
Data classes for pipeline configuration settings.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple
import torch


@dataclass
class ResizeConfig:
    """Configuration for output image resizing"""
    enabled: bool = True
    dimensions: Optional[Tuple[int, int]] = (64, 64)  # (width, height)
    scale: float = 1.0  # Only used if dimensions is None

    def __post_init__(self):
        if not self.enabled:
            self.dimensions = None
            self.scale = 1.0


@dataclass
class PipelineConfig:
    """Configuration for the upscaling pipeline"""
    models_dir: str = 'models/*'
    test_img_folder: str = 'LR/*'
    results_dir: str = 'all_models_results'
    specific_model: Optional[str] = "4x_foolhardy_Remacri" #None
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    resize: ResizeConfig = field(default_factory=ResizeConfig)