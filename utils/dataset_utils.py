"""
Dataset utilities for downloading and processing evaluation datasets.

This module provides utilities for downloading and processing the datasets.
"""
import json
import sys

from datasets import load_dataset
from pathlib import Path
from typing import List, Optional, Tuple
import re

from consts import *

# Compile all patterns into a single regex for efficiency
JB_REGEX = re.compile("|".join(JAILBREAK_PATTERNS), flags=re.IGNORECASE)


class DatasetDownloader:
    """Utility class for downloading and processing evaluation datasets."""

    def __init__(self, data_dir: str = "data"):
        """
        Initialize the dataset downloader.

        Args:
            data_dir: Directory to store downloaded datasets
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

    def download_huggingface_dataset(self, dataset_name: str,
                                     config: Optional[str] = None,
                                     split: str = "test") -> Path:
        """
        Download a dataset from Hugging Face Hub.

        Args:
            dataset_name: Name of the dataset on Hugging Face
            config: Configuration name (if applicable)
            split: Dataset split to download

        Returns:
            Path to the downloaded dataset file
        """
        try:
            from huggingface_hub import hf_hub_download

            filename = f"{dataset_name.replace('/', '_')}_{split}"
            local_path = self.data_dir / filename

            if local_path.exists():
                print(f"Dataset already exists at {local_path}")
                return local_path

            print(f"Downloading {dataset_name} from Hugging Face...")

            dataset = load_dataset(dataset_name, config, split=split)
            dataset.to_json(local_path)
            return local_path

        except ImportError:
            print("huggingface_hub not installed. Install with: pip install huggingface_hub", file=sys.stderr)
            raise

    @staticmethod
    def process_safeguard_dataset(dataset_path: Path) -> List[Tuple[str, int]]:
        """
        Process the safe-guard-prompt-injection dataset.

        Args:
            dataset_path: Path to the dataset file

        Returns:
            List of processed samples (limited to max_samples)
        """
        with open(dataset_path, 'r') as f:
            processed_samples = []
            for i, line in enumerate(f.readlines(), 1):
                line_data = line.strip()
                if not line_data:
                    continue
                try:
                    line_json = json.loads(line_data)
                    # Filter only the adversarial samples (label = 1) and filter out jailbreak samples
                    if line_json.get('label') == 1 and not DatasetDownloader._is_jailbreak(line_json.get('text')):
                        processed_samples.append((line_json.get('text'), 1))
                except json.JSONDecodeError as e:
                    print(f"[warn] skipping bad line {i}: {e}")

        print(f"Processed {len(processed_samples)} prompt injection samples")
        return processed_samples

    @staticmethod
    def download_research_datasets(data_dir: str = "data") -> Path:
        """
        Download datasets.

        Args:
            data_dir: Directory to store datasets

        Returns:
            prompt_injection_path
        """
        downloader = DatasetDownloader(data_dir)
        return downloader.download_huggingface_dataset("xTRam1/safe-guard-prompt-injection")

    @staticmethod
    def _is_jailbreak(text: str) -> bool:
        return bool(JB_REGEX.search(text))


if __name__ == "__main__":
    # Test dataset download
    pi_path = DatasetDownloader().download_research_datasets()
    print(f"Prompt injection dataset: {pi_path}")
