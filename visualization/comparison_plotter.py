"""
Model Comparison Plotter Module

This module provides functionality to create comparison charts for different models
based on their performance metrics (PSNR, SSIM, MSE) and processing times.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os
import re
from typing import Dict, List, Optional

# Set backend for matplotlib
matplotlib.use('TkAgg')


class ComparisonPlotter:
    """Creates comparison charts for multiple models."""

    def __init__(self, input_file: str, output_dir: str):
        """
        Initialize the comparison plotter.

        Args:
            input_file: Path to the model comparison summary text file
            output_dir: Directory where the output chart will be saved
        """
        self.input_file = input_file
        self.output_dir = output_dir
        self.output_file = os.path.join(output_dir, 'model_comparison_chart.png')

    def parse_results_file(self, file_path: str) -> Dict[str, List]:
        """
        Parse the model comparison results file.

        Args:
            file_path: Path to the results file

        Returns:
            Dictionary containing parsed model data

        Raises:
            FileNotFoundError: If the input file doesn't exist
        """
        models = []
        psnr = []
        ssim = []
        mse = []
        times = []

        with open(file_path, 'r') as f:
            lines = f.readlines()
            start_reading = False

            for line in lines:
                if line.startswith('----'):
                    start_reading = True
                    continue
                if not start_reading or not line.strip():
                    continue

                parts = re.split(r'\s{2,}', line.strip())
                if len(parts) >= 5:
                    models.append(parts[0].strip())
                    psnr.append(float(parts[1]))
                    ssim.append(float(parts[2]))
                    mse.append(float(parts[3]))
                    times.append(float(parts[4]))

        return {
            'Model Name': models,
            'PSNR (dB)': psnr,
            'SSIM': ssim,
            'MSE': mse,
            'Time (s)': times
        }

    def create_short_names(self, model_names: List[str], max_length: int = 15) -> List[str]:
        """
        Create shortened model names for better chart readability.

        Args:
            model_names: List of full model names
            max_length: Maximum length for shortened names

        Returns:
            List of shortened model names
        """
        short_names = []
        for name in model_names:
            # Split by underscore or dash and take first part
            short_name = re.split(r'_|-', name)[0]
            # Truncate if too long
            if len(short_name) > max_length:
                short_name = short_name[:max_length] + '...'
            short_names.append(short_name)
        return short_names

    def create_comparison_chart(self, df: pd.DataFrame, title: str = 'Model Comparison Metrics') -> None:
        """
        Create a 2x2 comparison chart showing all metrics.

        Args:
            df: DataFrame containing model comparison data
            title: Chart title
        """
        plt.figure(figsize=(18, 12))
        plt.suptitle(title, fontsize=16, y=1.02)

        # PSNR subplot
        plt.subplot(2, 2, 1)
        plt.barh(df['Short Name'], df['PSNR (dB)'], color='skyblue')
        plt.xlabel('PSNR (dB)')
        plt.title('Peak Signal-to-Noise Ratio (Higher is better)')
        plt.gca().invert_yaxis()

        # SSIM subplot
        plt.subplot(2, 2, 2)
        plt.barh(df['Short Name'], df['SSIM'], color='lightgreen')
        plt.xlabel('SSIM')
        plt.title('Structural Similarity (Higher is better)')
        plt.gca().invert_yaxis()

        # MSE subplot
        plt.subplot(2, 2, 3)
        plt.barh(df['Short Name'], df['MSE'], color='salmon')
        plt.xlabel('MSE')
        plt.title('Mean Squared Error (Lower is better)')
        plt.gca().invert_yaxis()

        # Processing time subplot
        plt.subplot(2, 2, 4)
        plt.barh(df['Short Name'], df['Time (s)'], color='gold')
        plt.xlabel('Time (seconds)')
        plt.title('Processing Time (Lower is better)')
        plt.gca().invert_yaxis()

        plt.tight_layout()

    def save_chart(self, dpi: int = 300) -> None:
        """
        Save the chart to the output file.

        Args:
            dpi: Resolution for the saved image
        """
        plt.savefig(self.output_file, bbox_inches='tight', dpi=dpi)
        plt.close()

    def plot_comparison(self, title: Optional[str] = None, show_chart: bool = False) -> bool:
        """
        Generate and save the complete model comparison chart.

        Args:
            title: Custom title for the chart
            show_chart: Whether to display the chart on screen

        Returns:
            True if successful, False otherwise
        """
        try:
            # Parse the results file
            data = self.parse_results_file(self.input_file)
            df = pd.DataFrame(data)

            # Create short names for better readability
            df['Short Name'] = self.create_short_names(df['Model Name'])

            # Create the chart
            chart_title = title or 'Model Comparison Metrics'
            self.create_comparison_chart(df, chart_title)

            # Save the chart
            self.save_chart()

            # Optionally show the chart
            if show_chart:
                plt.show()

            print(f'Diagram successfully saved: {self.output_file}')
            return True

        except FileNotFoundError:
            print(f'Error: File not found - {self.input_file}')
            return False
        except Exception as e:
            print(f'Error when processing data: {str(e)}')
            return False


def main():
    # Default paths (can be modified or made configurable)
    input_file = r'C:\Users\konte\ESRGAN\all_models_results\model_comparison_summary.txt'
    output_dir = r'C:\Users\konte\ESRGAN\all_models_results'

    # Create plotter and generate chart
    plotter = ComparisonPlotter(input_file, output_dir)
    plotter.plot_comparison(show_chart=True)


if __name__ == "__main__":
    main()