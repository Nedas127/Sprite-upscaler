"""
Model Processor
--------------
Handles the loading and processing of AI upscaling models.
"""

import glob
import json
import os
import shutil
import time
from typing import Any, Dict, Tuple

import numpy as np
import torch
from PIL import Image
from spandrel import ModelLoader

from config.config_models import PipelineConfig
from core.metrics_calculator import MetricsCalculator
from utils.image_utils import ImageUtils


class ModelProcessor:
    """Handles the loading and processing of models"""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.metrics_calculator = MetricsCalculator()

    def _load_model(self, model_path: str) -> Any:
        """Load model from file with error handling"""
        try:
            loader = ModelLoader(device=self.config.device)
            model = loader.load_from_file(model_path)
            model.eval()
            return model
        except Exception as e:
            raise RuntimeError(f"Error loading model {model_path}: {str(e)}")

    def _upscale_image(self, model: Any, img_tensor: torch.Tensor) -> np.ndarray:
        """Perform the actual upscaling with the model"""
        with torch.no_grad():
            output = model(img_tensor).squeeze(0).cpu().clamp(0, 1).numpy()
        return np.transpose(output[[2, 1, 0], :, :], (1, 2, 0)) * 255

    def upscale_with_model(self, model: Any, img_pil: Image.Image, bg_color: Tuple[int, int, int]) -> np.ndarray:
        """
        Upscale image on specified background color.

        Args:
            model: Loaded AI model
            img_pil: PIL Image to upscale
            bg_color: Background color as RGB tuple

        Returns:
            Upscaled image as RGB numpy array
        """
        bg = ImageUtils.prepare_background(img_pil, bg_color)
        img_np = np.array(bg).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(np.transpose(img_np[:, :, [2, 1, 0]], (2, 0, 1)))
        img_tensor = img_tensor.unsqueeze(0).to(self.config.device)
        return self._upscale_image(model, img_tensor).astype(np.uint8)

    def _process_single_image(self, model: Any, path: str, output_dir: str, model_name: str) -> Dict[str, Any]:
        """Process a single image with the model"""
        base = os.path.splitext(os.path.basename(path))[0]
        img_pil = Image.open(path)
        original_np = np.array(img_pil)

        # Upscale on white and black backgrounds
        white_bg = self.upscale_with_model(model, img_pil, (255, 255, 255))
        black_bg = self.upscale_with_model(model, img_pil, (0, 0, 0))

        # Extract transparency and true colors
        rgb, alpha = ImageUtils.extract_transparency(white_bg, black_bg)
        final_rgba = np.dstack((rgb, alpha))

        # Resize if needed
        if self.config.resize.enabled:
            final_rgba = ImageUtils.resize_image(
                final_rgba,
                dimensions=self.config.resize.dimensions,
                scale=self.config.resize.scale
            )

        # Save result
        output_path = os.path.join(output_dir, f'{base}_rlt.png')
        Image.fromarray(final_rgba, 'RGBA').save(output_path, quality=100, compress_level=0)

        # Calculate metrics
        metrics = self.metrics_calculator.calculate_metrics(original_np, final_rgba)
        return {
            'mse': metrics['mse'],
            'psnr': metrics['psnr'],
            'ssim': metrics['ssim']
        }

    def process_model(self, model_path: str) -> Dict[str, Any]:
        """Process all test images with a specific model."""
        print(f"\n{'=' * 80}")
        print(f"PROCESSING MODEL: {model_path}")
        print(f"{'=' * 80}")

        start_time = time.time()
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        output_dir = os.path.join(self.config.results_dir, f'{model_name}_results')

        # Setup output directory
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        # Initialize model
        try:
            model = self._load_model(model_path)
        except RuntimeError as e:
            return {
                "model_name": model_name,
                "error": str(e),
                "processing_time": 0,
                "metrics": {}
            }

        # Process all test images
        all_metrics = {}
        for idx, path in enumerate(glob.glob(self.config.test_img_folder), start=1):
            base = os.path.splitext(os.path.basename(path))[0]
            print(f"{idx}: Processing {base}")

            try:
                metrics = self._process_single_image(model, path, output_dir, model_name)
                all_metrics[base] = metrics
                print(f"  MSE:  {metrics['mse']:.6f}")
                print(f"  PSNR: {metrics['psnr']:.6f} dB")
                print(f"  SSIM: {metrics['ssim']:.6f}")
            except Exception as e:
                print(f"Error processing {path}: {str(e)}")
                all_metrics[base] = {'error': str(e)}

        return self._generate_model_summary(model_name, all_metrics, start_time, output_dir)

    def _generate_model_summary(self, model_name: str, all_metrics: Dict[str, Any],
                                start_time: float, output_dir: str) -> Dict[str, Any]:
        """Generate summary statistics for the model run."""
        valid_metrics = [m for m in all_metrics.values() if 'error' not in m]
        model_summary = {
            "model_name": model_name,
            "processing_time": time.time() - start_time,
            "metrics": {}
        }

        if valid_metrics:
            avg_mse = sum(m['mse'] for m in valid_metrics) / len(valid_metrics)
            psnr_values = [m['psnr'] for m in valid_metrics]
            finite_psnr_values = [p for p in psnr_values if p != float('inf')]
            avg_psnr = sum(finite_psnr_values) / len(finite_psnr_values) if finite_psnr_values else 100.0
            avg_ssim = sum(m['ssim'] for m in valid_metrics) / len(valid_metrics)

            all_metrics["_average"] = {
                'avg_mse': avg_mse,
                'avg_psnr': avg_psnr,
                'avg_ssim': avg_ssim
            }

            model_summary["metrics"] = {
                'avg_mse': avg_mse,
                'avg_psnr': avg_psnr,
                'avg_ssim': avg_ssim,
                'processed_images': len(valid_metrics),
                'total_images': len(glob.glob(self.config.test_img_folder))
            }

            print("\nAverage metrics across all images:")
            print(f"  Avg MSE:  {avg_mse:.6f}")
            print(f"  Avg PSNR: {avg_psnr:.6f} dB")
            print(f"  Avg SSIM: {avg_ssim:.6f}")

        # Save metrics to file
        metrics_file = os.path.join(output_dir, f'{model_name}_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(all_metrics, f, indent=2)

        print(f"\nAll metrics for {model_name} saved to {metrics_file}")
        print(f"Total processing time: {model_summary['processing_time']:.2f} seconds")
        return model_summary
