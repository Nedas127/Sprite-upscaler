"""
Image Upscaling Pipeline with Transparency Handling
-------------------------------------------------
This module provides a framework for upscaling images with multiple AI models
while preserving transparency through dual background technique.
"""

import glob
import os
import time
import sys
import json
import shutil
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Union, Any

import cv2
import numpy as np
import torch
from PIL import Image
from spandrel import ModelLoader

from image_metrics import compute_metrics

# Type aliases for clarity
ImageArray = np.ndarray  # numpy array representing an image
RGBArray = np.ndarray  # RGB image as numpy array
RGBAArray = np.ndarray  # RGBA image as numpy array


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
    specific_model: Optional[str] = "4x_PixelPerfectV4_137000_G" #None
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    resize: ResizeConfig = field(default_factory=ResizeConfig)


class ImageProcessor:
    """Handles image processing operations for the upscaling pipeline"""

    @staticmethod
    def resize_image(image: RGBAArray, config: ResizeConfig) -> RGBAArray:
        """
        Resize image based on the configured resize settings.

        Args:
            image: RGBA numpy array
            config: Resize configuration

        Returns:
            Resized RGBA numpy array
        """
        if not config.enabled:
            return image

        # Create PIL Image for better quality resizing
        pil_img = Image.fromarray(image)

        if config.dimensions:
            # Resize to specific dimensions
            resized = pil_img.resize(config.dimensions, Image.LANCZOS)
        else:
            # Resize by scale factor
            new_width = int(image.shape[1] * config.scale)
            new_height = int(image.shape[0] * config.scale)
            resized = pil_img.resize((new_width, new_height), Image.LANCZOS)

        # Convert back to numpy array
        return np.array(resized)

    @staticmethod
    def extract_transparency(white_bg: RGBArray, black_bg: RGBArray) -> Tuple[RGBArray, np.ndarray]:
        """
        Create alpha channel from differences between white/black backgrounds

        Args:
            white_bg: RGB image upscaled on white background
            black_bg: RGB image upscaled on black background

        Returns:
            Tuple of (RGB array, alpha channel array)
        """
        # Convert to float for precise calculations
        white = white_bg.astype(np.float32)
        black = black_bg.astype(np.float32)

        # Calculate alpha (transparency)
        with np.errstate(divide='ignore', invalid='ignore'):
            diff = white - black
            alpha = 1.0 - np.clip(np.mean(diff, axis=2) / 255.0, 0, 1)
            alpha = (alpha * 255).clip(0, 255).astype(np.uint8)

            # Calculate true RGB values
            alpha_float = np.where(alpha > 0, alpha / 255.0, 1.0)
            rgb = (black / alpha_float[..., None]).clip(0, 255).astype(np.uint8)

        return rgb, alpha

    @staticmethod
    def post_process_alpha(alpha: np.ndarray) -> np.ndarray:
        """
        Clean up alpha channel with edge refinement

        Args:
            alpha: Raw alpha channel array

        Returns:
            Refined alpha channel array
        """
        # Remove noise
        alpha = cv2.medianBlur(alpha, 3)

        # Enhance edges
        edges = cv2.Canny(alpha, 50, 150)
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

        # Apply edge-aware smoothing (with fallback)
        try:
            if hasattr(cv2, 'ximgproc'):
                guided_filter = cv2.ximgproc.createGuidedFilter(alpha, radius=5, eps=0.01)
                alpha = guided_filter.filter(alpha)
            else:
                alpha = cv2.bilateralFilter(alpha, 5, 75, 75)
        except Exception:
            alpha = cv2.bilateralFilter(alpha, 5, 75, 75)

        return alpha


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
        if original_img.shape[2] == 4:  # RGBA
            original_rgb = original_img[:, :, :3]
        else:
            original_rgb = original_img

        if upscaled_img.shape[2] == 4:  # RGBA
            upscaled_rgb = upscaled_img[:, :, :3]
        else:
            upscaled_rgb = upscaled_img

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
            # Set a high but finite value for PSNR when MSE is zero
            metrics['psnr'] = 100.0  # 100 dB is extremely good quality

        return metrics


class ModelProcessor:
    """Handles the loading and processing of models"""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.image_processor = ImageProcessor()
        self.metrics_calculator = MetricsCalculator()

    def upscale_with_model(self, model: Any, img_pil: Image.Image, bg_color: Tuple[int, int, int]) -> RGBArray:
        """
        Upscale image on specified background color

        Args:
            model: Loaded AI model
            img_pil: PIL Image to upscale
            bg_color: Background color as RGB tuple

        Returns:
            Upscaled image as RGB numpy array
        """
        # Create background
        bg = Image.new('RGB', img_pil.size, bg_color)
        if img_pil.mode == 'RGBA':
            bg.paste(img_pil, (0, 0), img_pil)
        else:
            bg.paste(img_pil, (0, 0))

        # Convert to tensor
        img_np = np.array(bg).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(np.transpose(img_np[:, :, [2, 1, 0]], (2, 0, 1)))
        img_tensor = img_tensor.unsqueeze(0).to(self.config.device)

        # Upscale
        with torch.no_grad():
            output = model(img_tensor).squeeze(0).cpu().clamp(0, 1).numpy()

        # Convert back to numpy (RGB)
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        return (output * 255).astype(np.uint8)

    def process_model(self, model_path: str) -> Dict[str, Any]:
        """
        Process all test images with a specific model

        Args:
            model_path: Path to the model file

        Returns:
            Dictionary with model processing summary
        """
        print(f"\n{'=' * 80}")
        print(f"PROCESSING MODEL: {model_path}")
        print(f"{'=' * 80}")

        start_time = time.time()

        # Dynamically create output directory name from model filename
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        output_dir = os.path.join(self.config.results_dir, f'{model_name}_results')

        # Delete and recreate the output directory if it already exists
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        # Metrics file inside output directory
        metrics_file = os.path.join(output_dir, f'{model_name}_metrics.json')

        # Initialize model
        try:
            loader = ModelLoader(device=self.config.device)
            model = loader.load_from_file(model_path)
            model.eval()
        except Exception as e:
            print(f"Error loading model {model_path}: {str(e)}")
            return {
                "model_name": model_name,
                "error": str(e),
                "processing_time": 0,
                "metrics": {}
            }

        # Dictionary to store metrics for all processed images
        all_metrics = {}

        # Process all test images
        for idx, path in enumerate(glob.glob(self.config.test_img_folder), start=1):
            base = os.path.splitext(os.path.basename(path))[0]
            print(f"{idx}: Processing {base}")

            try:
                # Load original image
                img_pil = Image.open(path)
                original_np = np.array(img_pil)

                # Upscale on white and black backgrounds
                white_bg = self.upscale_with_model(model, img_pil, (255, 255, 255))
                black_bg = self.upscale_with_model(model, img_pil, (0, 0, 0))

                # Extract transparency and true colors
                rgb, alpha = self.image_processor.extract_transparency(white_bg, black_bg)

                # Refine alpha channel
                #alpha = self.image_processor.post_process_alpha(alpha)

                # Recombine with processed alpha
                final_rgba = np.dstack((rgb, alpha))

                # Resize if needed
                if self.config.resize.enabled:
                    final_rgba = self.image_processor.resize_image(final_rgba, self.config.resize)

                final = Image.fromarray(final_rgba, 'RGBA')

                # Save result
                output_path = os.path.join(output_dir, f'{base}_rlt.png')
                final.save(output_path, quality=100, compress_level=0)

                # Calculate and store image quality metrics
                metrics = self.metrics_calculator.calculate_metrics(original_np, final_rgba)
                all_metrics[base] = {
                    'mse': metrics['mse'],
                    'psnr': metrics['psnr'],
                    'ssim': metrics['ssim']
                }

                # Print metrics for this image
                print(f"  MSE:  {metrics['mse']:.6f}")
                print(f"  PSNR: {metrics['psnr']:.6f} dB")
                print(f"  SSIM: {metrics['ssim']:.6f}")

            except Exception as e:
                print(f"Error processing {path} with {model_name}: {str(e)}")
                all_metrics[base] = {'error': str(e)}

        # Calculate and store average metrics across all images
        valid_metrics = [m for m in all_metrics.values() if 'error' not in m]
        model_summary = {
            "model_name": model_name,
            "processing_time": time.time() - start_time,
            "metrics": {}
        }

        if valid_metrics:
            avg_mse = sum(m['mse'] for m in valid_metrics) / len(valid_metrics)

            # Calculate average PSNR properly, handling infinity values
            psnr_values = [m['psnr'] for m in valid_metrics]
            finite_psnr_values = [p for p in psnr_values if p != float('inf')]

            if finite_psnr_values:
                # If there are some finite values, average those
                avg_psnr = sum(finite_psnr_values) / len(finite_psnr_values)
            else:
                # If all values are infinity, set a high finite value
                avg_psnr = 100.0

            avg_ssim = sum(m['ssim'] for m in valid_metrics) / len(valid_metrics)

            # Add average metrics to the JSON
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

        # Save all metrics (including averages) to a JSON file
        with open(metrics_file, 'w') as f:
            json.dump(all_metrics, f, indent=2)

        print(f"\nAll metrics for {model_name} saved to {metrics_file}")
        print(f"Total processing time: {model_summary['processing_time']:.2f} seconds")

        return model_summary


class UpscalingPipeline:
    """Main pipeline for processing multiple models and generating comparisons"""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.model_processor = ModelProcessor(config)

        # Create main results directory
        if not os.path.exists(config.results_dir):
            os.makedirs(config.results_dir)

        self.summary_file = os.path.join(config.results_dir, 'models_comparison.json')

    def run(self) -> None:
        """Run the complete upscaling pipeline for all models"""
        print(f"Using device: {self.config.device}")
        print(f"Looking for models in: {self.config.models_dir}")
        print(f"Test images: {self.config.test_img_folder}")

        all_models = glob.glob(self.config.models_dir)
        print(f"Found {len(all_models)} models to process")

        # Dictionary to store summaries for all models
        all_model_summaries = []

        # Process each model
        for model_path in all_models:
            if os.path.isfile(model_path) and any(model_path.endswith(ext) for ext in ['.pth', '.pt', '.ckpt']):
                # Skip models that don't match specific_model if it's set
                if self.config.specific_model and self.config.specific_model not in model_path:
                    continue

                model_summary = self.model_processor.process_model(model_path)
                all_model_summaries.append(model_summary)

        # Save summary comparison file
        with open(self.summary_file, 'w') as f:
            json.dump(all_model_summaries, f, indent=2)

        # Print comparison table
        self._print_comparison_table(all_model_summaries)

    def _print_comparison_table(self, summaries: List[Dict[str, Any]]) -> None:
        """Print a comparison table of all model results"""
        print("\n\n")
        print("MODEL COMPARISON SUMMARY")
        print("=" * 80)
        print(f"{'Model Name':<40} {'PSNR (dB)':<10} {'SSIM':<10} {'MSE':<10} {'Time (s)':<10}")
        print("-" * 80)

        # Sort by PSNR (higher is better)
        summaries.sort(key=lambda x: x.get("metrics", {}).get("avg_psnr", 0), reverse=True)

        for summary in summaries:
            model_name = summary["model_name"]
            metrics = summary.get("metrics", {})

            if "error" in summary:
                print(f"{model_name:<40} ERROR: {summary['error']}")
            else:
                psnr = metrics.get("avg_psnr", 0)
                ssim = metrics.get("avg_ssim", 0)
                mse = metrics.get("avg_mse", 0)
                time_taken = summary["processing_time"]

                print(f"{model_name:<40} {psnr:<10.4f} {ssim:<10.4f} {mse:<10.4f} {time_taken:<10.2f}")

        print("\nDetailed comparison saved to:", self.summary_file)


def parse_args() -> PipelineConfig:
    """Parse command line arguments and return a config object"""
    config = PipelineConfig()

    if len(sys.argv) > 1:
        # First argument could be either model name or resize parameters
        if sys.argv[1].startswith("--resize="):
            # Handle resize parameter
            resize_str = sys.argv[1].split("=")[1]
            config.resize.enabled = True

            # Check if it's dimensions or scale
            if "x" in resize_str:
                # Format: --resize=64x64
                width, height = map(int, resize_str.split("x"))
                config.resize.dimensions = (width, height)
                config.resize.scale = None
                print(f"Command line resize: output will be {width}x{height} pixels")
            else:
                # Format: --resize=0.5 or --resize=50
                try:
                    scale = float(resize_str)
                    if scale > 1:  # Assume it's a percentage if > 1
                        scale = scale / 100.0
                    config.resize.scale = scale
                    config.resize.dimensions = None
                    print(f"Command line resize: output will be scaled to {int(scale * 100)}%")
                except ValueError:
                    print(f"Invalid resize parameter: {resize_str}")
                    print("Format should be --resize=WIDTHxHEIGHT or --resize=SCALE")
                    sys.exit(1)

            # Check if there's a second argument for model name
            if len(sys.argv) > 2 and not sys.argv[2].startswith("--"):
                config.specific_model = sys.argv[2]
                print(f"Command line model: {config.specific_model}")
        else:
            # First argument is the model name
            config.specific_model = sys.argv[1]
            print(f"Command line model: {config.specific_model}")

            # Check if there's a second argument for resize
            if len(sys.argv) > 2 and sys.argv[2].startswith("--resize="):
                resize_str = sys.argv[2].split("=")[1]
                config.resize.enabled = True

                # Parse resize parameter
                if "x" in resize_str:
                    width, height = map(int, resize_str.split("x"))
                    config.resize.dimensions = (width, height)
                    config.resize.scale = None
                    print(f"Command line resize: output will be {width}x{height} pixels")
                else:
                    try:
                        scale = float(resize_str)
                        if scale > 1:  # Assume it's a percentage if > 1
                            scale = scale / 100.0
                        config.resize.scale = scale
                        config.resize.dimensions = None
                        print(f"Command line resize: output will be scaled to {int(scale * 100)}%")
                    except ValueError:
                        print(f"Invalid resize parameter: {resize_str}")
                        print("Format should be --resize=WIDTHxHEIGHT or --resize=SCALE")
                        sys.exit(1)

    return config


def main():
    """Main entry point for the application"""
    config = parse_args()
    pipeline = UpscalingPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()