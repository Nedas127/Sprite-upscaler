"""
Upscaling Pipeline
-----------------
Main pipeline for processing multiple models and generating comparisons.
"""

import glob
import json
import os
from typing import Any, Dict, List

from config.config_models import PipelineConfig
from core.model_processor import ModelProcessor


class UpscalingPipeline:
    """Main pipeline for processing multiple models and generating comparisons"""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.model_processor = ModelProcessor(config)

        # Create main results directory
        os.makedirs(config.results_dir, exist_ok=True)
        self.summary_file = os.path.join(config.results_dir, 'models_comparison.json')

    def run(self) -> None:
        """Run the complete upscaling pipeline for all models"""
        print(f"Using device: {self.config.device}")
        print(f"Looking for models in: {self.config.models_dir}")
        print(f"Test images: {self.config.test_img_folder}")

        all_models = glob.glob(self.config.models_dir)
        print(f"Found {len(all_models)} models to process")

        all_model_summaries = []
        for model_path in all_models:
            if os.path.isfile(model_path) and any(model_path.endswith(ext) for ext in ['.pth', '.pt', '.ckpt']):
                if self.config.specific_model and self.config.specific_model not in model_path:
                    continue
                all_model_summaries.append(self.model_processor.process_model(model_path))

        self._save_and_display_results(all_model_summaries)

    def _save_and_display_results(self, summaries: List[Dict[str, Any]]) -> None:
        """Save results and display comparison table"""
        with open(self.summary_file, 'w') as f:
            json.dump(summaries, f, indent=2)

        self._print_comparison_table(summaries)

    def _print_comparison_table(self, summaries: List[Dict[str, Any]]) -> None:
        """Print a comparison table of all model results"""
        print("\n\nMODEL COMPARISON SUMMARY")
        print("=" * 80)
        print(f"{'Model Name':<40} {'PSNR (dB)':<10} {'SSIM':<10} {'MSE':<10} {'Time (s)':<10}")
        print("-" * 80)

        summaries.sort(key=lambda x: x.get("metrics", {}).get("avg_psnr", 0), reverse=True)

        for summary in summaries:
            model_name = summary["model_name"]
            if "error" in summary:
                print(f"{model_name:<40} ERROR: {summary['error']}")
                continue

            metrics = summary.get("metrics", {})
            print(f"{model_name:<40} "
                  f"{metrics.get('avg_psnr', 0):<10.4f} "
                  f"{metrics.get('avg_ssim', 0):<10.4f} "
                  f"{metrics.get('avg_mse', 0):<10.4f} "
                  f"{summary['processing_time']:<10.2f}")

        print("\nDetailed comparison saved to:", self.summary_file)
