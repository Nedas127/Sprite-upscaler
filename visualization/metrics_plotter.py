"""
Metrics Plotter Module

This module provides functionality to create detailed metric visualization charts
for individual models, showing PSNR, SSIM, and MSE values across different test images.
"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import json
import pandas as pd
from typing import Dict, Optional, Tuple

# Set backend for matplotlib
matplotlib.use('TkAgg')


class MetricsPlotter:
    """Creates detailed metrics visualization charts for ESRGAN model performance."""

    def __init__(self, model_name: str, base_dir: str):
        """
        Initialize the metrics plotter.

        Args:
            model_name: Name of the model to visualize
            base_dir: Base directory containing model results
        """
        self.model_name = model_name
        self.base_dir = base_dir
        self.model_folder = os.path.join(base_dir, model_name + "_results")
        self.json_file = os.path.join(self.model_folder, model_name + "_metrics.json")
        self.output_path = os.path.join(self.model_folder, "metrics_diagram.png")

        # Color scheme for the charts
        self.colors = {
            'psnr': '#FF8000',  # Dark orange
            'ssim': '#00A64C',  # Dark green
            'mse': '#0066CC',  # Dark blue
            'background': '#F5F5F5',  # Light gray
            'grid': '#E0E0E0'  # Lighter gray for grid
        }

    def load_metrics_data(self) -> pd.DataFrame:
        """
        Load and prepare metrics data from JSON file.

        Returns:
            DataFrame with metrics data, sorted by image ID

        Raises:
            FileNotFoundError: If the metrics JSON file doesn't exist
            json.JSONDecodeError: If the JSON file is malformed
        """
        with open(self.json_file, 'r') as f:
            data = json.load(f)

        # Remove average data if present
        if "_average" in data:
            data.pop("_average")

        # Convert to DataFrame and sort
        df = pd.DataFrame(data).T
        df.index = df.index.astype(str)
        df = df.sort_index()

        return df

    def setup_subplot(self, ax: plt.Axes, metric: str, df: pd.DataFrame,
                      subplot_index: int, total_subplots: int) -> None:
        """
        Configure a single subplot for a specific metric.

        Args:
            ax: Matplotlib axes object
            metric: Metric name ('psnr', 'ssim', or 'mse')
            df: DataFrame containing the metrics data
            subplot_index: Current subplot index (0-based)
            total_subplots: Total number of subplots
        """
        # Define marker styles for each metric
        markers = {'psnr': 'o', 'ssim': 's', 'mse': '^'}

        # Plot the metric line
        x_range = range(len(df.index))
        ax.plot(x_range, df[metric],
                color=self.colors[metric],
                marker=markers[metric],
                linewidth=2,
                markersize=6)

        # Configure axes
        ax.set_ylabel(metric.upper(), fontsize=12, fontweight='bold',
                      color=self.colors[metric])
        ax.tick_params(axis='y', labelcolor=self.colors[metric])

        # Set title only for the first subplot
        if subplot_index == 0:
            ax.set_title(f"{self.model_name} metrikų diagrama",
                         fontsize=14, fontweight='bold', pad=20)

        # Set xlabel only for the last subplot
        if subplot_index == total_subplots - 1:
            ax.set_xlabel("Vaizdo ID", fontsize=12, fontweight='bold')

        # Style the subplot
        ax.set_facecolor(self.colors['background'])
        ax.grid(True, linestyle='--', alpha=0.7, color=self.colors['grid'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Set x-axis ticks and labels
        ax.set_xticks(x_range)
        ax.set_xticklabels(df.index, rotation=45, ha='right', fontsize=9)

        # Add value annotations above points
        precision = 3 if metric == 'mse' else 2
        for i, value in enumerate(df[metric]):
            ax.annotate(f"{value:.{precision}f}", (i, value),
                        textcoords="offset points", xytext=(0, 5),
                        ha='center', fontsize=8, color=self.colors[metric])

    def create_metrics_chart(self, df: pd.DataFrame, figsize: Tuple[int, int] = (12, 10)) -> None:
        """
        Create the complete metrics visualization chart.

        Args:
            df: DataFrame containing metrics data
            figsize: Figure size as (width, height) tuple
        """
        plt.figure(figsize=figsize)

        # Create grid layout with 3 subplots
        gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1], hspace=0.3)

        metrics = ['psnr', 'ssim', 'mse']

        # Create each subplot
        for i, metric in enumerate(metrics):
            ax = plt.subplot(gs[i])
            self.setup_subplot(ax, metric, df, i, len(metrics))

            # Share x-axis for better alignment
            if i > 0:
                ax.sharex(plt.subplot(gs[0]))

    def save_chart(self, dpi: int = 300) -> None:
        """
        Save the chart to the output file.

        Args:
            dpi: Resolution for the saved image
        """
        plt.tight_layout(pad=2.0)
        plt.savefig(self.output_path, dpi=dpi, bbox_inches='tight')

    def plot_metrics(self, show_chart: bool = True, save_chart: bool = True,
                     figsize: Optional[Tuple[int, int]] = None) -> bool:
        """
        Generate the complete metrics visualization.

        Args:
            show_chart: Whether to display the chart on screen
            save_chart: Whether to save the chart to file
            figsize: Custom figure size as (width, height) tuple

        Returns:
            True if successful, False otherwise
        """
        try:
            # Load the metrics data
            df = self.load_metrics_data()

            if df.empty:
                print(f"No metrics data found for model: {self.model_name}")
                return False

            # Create the chart
            chart_size = figsize or (12, 10)
            self.create_metrics_chart(df, chart_size)

            # Save the chart if requested
            if save_chart:
                self.save_chart()
                print(f"Metrics diagram saved to: {self.output_path}")

            # Show the chart if requested
            if show_chart:
                plt.show()
            else:
                plt.close()

            return True

        except FileNotFoundError:
            print(f"Error: Metrics file not found - {self.json_file}")
            return False
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format in - {self.json_file}")
            return False
        except Exception as e:
            print(f"Error creating metrics plot: {str(e)}")
            return False

    def get_metrics_summary(self) -> Optional[Dict]:
        """
        Get a summary of the metrics data.

        Returns:
            Dictionary containing metrics summary or None if error
        """
        try:
            df = self.load_metrics_data()

            summary = {}
            for metric in ['psnr', 'ssim', 'mse']:
                summary[metric] = {
                    'mean': float(df[metric].mean()),
                    'std': float(df[metric].std()),
                    'min': float(df[metric].min()),
                    'max': float(df[metric].max()),
                    'count': len(df[metric])
                }

            return summary

        except Exception as e:
            print(f"Error getting metrics summary: {str(e)}")
            return None


def main():
    # Default configuration (can be made configurable via command line args)
    model_name = "4x_foolhardy_Remacri"
    base_dir = r"C:\Users\konte\ESRGAN\all_models_results"

    # Create plotter and generate chart
    plotter = MetricsPlotter(model_name, base_dir)

    # Generate the metrics plot
    success = plotter.plot_metrics(show_chart=True)

    if success:
        # Optionally print summary statistics
        summary = plotter.get_metrics_summary()
        if summary:
            print("\nMetrics Summary:")
            for metric, stats in summary.items():
                print(f"{metric.upper()}: Mean={stats['mean']:.3f}, "
                      f"Std={stats['std']:.3f}, Range=[{stats['min']:.3f}, {stats['max']:.3f}]")


if __name__ == "__main__":
    main()