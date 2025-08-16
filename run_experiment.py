from datetime import datetime

from pathlib import Path
from typing import Tuple, List

from attacker.aml_attacker import AmlAttacker, AMLTechnique
from guard_adapters.guardrail_adapter import GuardResult
from utils.dataset_utils import DatasetDownloader
from guard_adapters.llm_guard_adapter import LLMGuardAdapter


DATASET_PATH = Path("utils/data/xTRam1_safe-guard-prompt-injection_test")


def run_experiment():
    start_time = datetime.now()
    print(f"start time: {start_time}")

    # Adjust here the argument False/True to control whether to use the defense amplifier
    llm_guard_adapter = LLMGuardAdapter(True)
    aml_attacker = AmlAttacker()

    data: List[Tuple[str, int]] = DatasetDownloader.process_safeguard_dataset(DATASET_PATH)
    print(f"Processed {len(data)} samples")

    # First filter out all the prompt injection that are misclassified without additional attack
    data_for_attacker = []
    for sample in data:
        result: GuardResult = llm_guard_adapter.check(sample[0])
        if not result.allowed:
            data_for_attacker.append(sample)
    misclassified_data_count = len(data) - len(data_for_attacker)
    print(f"Misclassified samples: {misclassified_data_count} from {len(data)}")

    # Attack
    print(f"Processed {len(data_for_attacker)} samples for attacker")

    # Adjust here the AMLTechnique for testing different techniques
    successful_attacks_count = aml_attacker.attack(data_for_attacker, AMLTechnique.PWWS, llm_guard_adapter)
    print(f"Attack succeeded on {successful_attacks_count} from {len(data_for_attacker)}")

    # Calculate ASR - attack success rate
    print(f"ASR: {(successful_attacks_count + misclassified_data_count) / len(data)}")

    end_time = datetime.now()
    minutes = (end_time - start_time).total_seconds() / 60
    print(f"end time: {end_time}")
    print(f"TOTAL TIME: {minutes} minutes")


if __name__ == '__main__':
    run_experiment()
