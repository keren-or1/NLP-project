"""
Visualization utilities for adversarial attack evaluation results.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import warnings

warnings.filterwarnings('ignore', category=UserWarning)


class AttackVisualizationUtils:
    """Utility class for generating attack evaluation visualizations."""

    def __init__(self, style: str = 'default', figsize: Tuple[int, int] = (15, 10)):
        """
        Initialize visualization utilities.
        
        Args:
            style: Matplotlib style to use
            figsize: Default figure size
        """
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')
        self.default_figsize = figsize
        self.colors = {
            'baseline': '#ff7f7f',
            'amplifier': '#7fbf7f',
            'textfooler': '#ff7f7f',
            'alzantot': '#7fbf7f',
            'deepwordbug': '#7f7fff',
            'pwws': '#ffbf7f'
        }
        self.results_data = AttackVisualizationUtils._load_default_results_data()

    def generate_report_plots(self,
                              output_dir: str = "plots",
                              show: bool = False) -> Dict[str, plt.Figure]:
        """
        Generate report standard plots from results data.

        Args:
            output_dir: Directory to save plots
            show: Whether to display plots

        Returns:
            Dictionary of plot names to Figure objects
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        plot_figures = {}

        print(f"Generating plots in directory: {output_path}")
        plot_figures['asr_comparison'] = self.plot_asr_comparison(
            self.results_data['attack_methods'],
            self.results_data['baseline_asr'],
            self.results_data['amplifier_asr'],
            save_path=str(output_path / 'asr_comparison.png'),
            show=show
        )

        return plot_figures

    def plot_asr_comparison(self,
                            attack_methods: List[str],
                            baseline_asr: List[float],
                            amplifier_asr: List[float],
                            save_path: Optional[str] = None,
                            show: bool = False) -> plt.Figure:
        """
        Create ASR comparison plot between baseline and defense amplifier.
        
        Args:
            attack_methods: List of attack method names
            baseline_asr: List of baseline ASR values
            amplifier_asr: List of amplifier ASR values
            save_path: Path to save the plot (optional)
            show: Whether to display the plot
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(1, 1, figsize=self.default_figsize, constrained_layout=True)

        x = np.arange(len(attack_methods))
        width = 0.2

        bars1 = ax.bar(x - width / 2, baseline_asr, width,
                       label='Baseline', color=self.colors['baseline'], alpha=0.8)
        bars2 = ax.bar(x + width / 2, amplifier_asr, width,
                       label='Defense Amplifier', color=self.colors['amplifier'], alpha=0.8)

        ax.set_xlabel('Attack Method', fontsize=18)
        ax.set_ylabel('Attack Success Rate (ASR)', fontsize=18)
        ax.set_title('Attack Success Rates: Baseline vs Defense Amplifier', fontsize=22, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(attack_methods, rotation=45, ha='right', fontsize=16)
        ax.legend(fontsize=15)
        ax.grid(True, alpha=0.3)

        AttackVisualizationUtils._add_bar_labels(bars1, ax, format_str='{:.3f}')
        AttackVisualizationUtils._add_bar_labels(bars2, ax, format_str='{:.3f}')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ASR comparison plot saved to: {save_path}")

        if show:
            try:
                plt.show()
            except (AttributeError, Exception) as e:
                print(f"Display not available (running in non-interactive environment): {e}")
                print("Plot generated successfully - check saved file if save_path was provided.")

        return fig

    @staticmethod
    def _add_bar_labels(bars, ax, format_str: str = '{:.2f}', offset: float = 0.001):
        """Add value labels on top of bars."""
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + offset,
                    format_str.format(height), ha='center', va='bottom', fontsize=20)

    @staticmethod
    def _load_default_results_data() -> Dict:
        """Load the default experimental results data."""
        return {
            'attack_methods': ['TextFooler', 'Alzantot', 'DeepWordBug', 'PWWS'],
            'baseline_asr': [0.413, 0.280, 0.398, 0.339],
            'amplifier_asr': [0.210, 0.210, 0.210, 0.210],
            'baseline_times': [13.69, 8.36, 14.38, 14.14],
            'amplifier_times': [76.87, 25.28, 70.90, 108.54],
            'successful_attacks': [96, 33, 89, 61]
        }


if __name__ == "__main__":
    print("Generating plots...")
    viz = AttackVisualizationUtils()

    figures = viz.generate_report_plots(show=False)
    print(f"Generated {len(figures)} plots successfully!")
    print("Files created in 'plots/' directory:")
    for plot_name in figures.keys():
        print(f"  - {plot_name}.png")
