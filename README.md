# NLP Security Research Project

A comprehensive framework for testing adversarial attacks against LLM guards and prompt injection detection systems. This project implements various attack techniques and defense mechanisms to evaluate the robustness of AI safety systems.

## 🎯 Project Overview

This project focuses on:
- **Adversarial Machine Learning (AML) Attacks**: Testing various attack techniques against LLM guards
- **Prompt Injection Detection**: Evaluating defense mechanisms against malicious prompts
- **Defense Amplification**: Implementing enhanced security measures
- **Attack Success Rate (ASR) Analysis**: Measuring the effectiveness of attacks and defenses

## 🏗️ Project Structure

```
nlp-project/
├── attacker/                      # Attack implementation modules
│   ├── aml_attacker.py            # Main attacker class with various AML techniques
│   └── guardrail_model_wrapper.py # Model wrapper for guardrail integration
├── guard_adapters/                # Guard system adapters
│   ├── llm_guard_adapter.py       # LLM Guard integration
│   └── guardrail_adapter.py       # Base guardrail interface
├── defense_amplifier/             # Defense enhancement modules
│   └── defense_amplifier.py       # Defense amplification logic
├── utils/                         # Utility functions
│   ├── dataset_utils.py           # Dataset processing utilities
│   ├── visualization_utils.py     # Result visualization tools
│   └── data/                      # Dataset storage directory
├── consts.py                      # Jailbreak patterns and constants
├── run_experiment.py              # Main experiment runner
└── requirements.txt               # Project dependencies
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd nlp-project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Dependencies

The project uses the following key libraries:
- `llm-guard`: LLM security and safety framework
- `textattack`: Adversarial attack framework for NLP
- `huggingface_hub`: Hugging Face model integration
- `datasets`: Dataset loading and processing

## 🔬 Usage

### Running Experiments

Execute the main experiment script:

```bash
python run_experiment.py
```

### Configuration Options

The experiment can be configured by modifying parameters in `run_experiment.py`:

1. **Defense Amplifier**: Toggle defense amplification on/off
```python
llm_guard_adapter = LLMGuardAdapter(True)  # True to enable defense amplifier
```

2. **Attack Technique**: Choose from available AML techniques
```python
attack_results = aml_attacker.attack(data_for_attacker, AMLTechnique.PWWS, llm_guard_adapter)
```

### Available Attack Techniques

The project supports various AML attack techniques through the `AMLTechnique` enum:
- `TEXTFOOLER`: TextFooler attack using word importance ranking and synonym replacement
- `ALZANTOT`: Genetic algorithm-based attack with population-based search
- `DEEPWORDBUG`: Character-level perturbations targeting deep learning models
- `PWWS`: Probability Weighted Word Saliency attack using word importance scores

## 📊 Experiment Flow

1. **Data Loading**: Process the safeguard dataset from `utils/data/xTRam1_safe-guard-prompt-injection_test`
2. **Initial Filtering**: Identify samples that are already misclassified by the guard
3. **Attack Execution**: Apply selected attack techniques to remaining samples
4. **Results Analysis**: Calculate Attack Success Rate (ASR) and other metrics
5. **Performance Reporting**: Display timing and success statistics

## 📈 Metrics

The system evaluates:
- **Attack Success Rate (ASR)**: Percentage of successful attacks
- **Misclassification Rate**: Baseline failure rate without attacks
- **Processing Time**: Performance benchmarks

## 🔧 Customization

### Adding New Attack Techniques

1. Extend the `AMLTechnique` enum in the attacker module
2. Implement the attack logic in `AmlAttacker` class
3. Update the experiment configuration

### Implementing Custom Guards

1. Create a new adapter in `guard_adapters/`
2. Inherit from the base guardrail interface
3. Implement the `check()` method returning `GuardResult`

### Dataset Integration

The project uses the xTRam1 safeguard dataset. To use custom datasets:
1. Update the `DATASET_PATH` in `run_experiment.py`
2. Modify `DatasetDownloader.process_safeguard_dataset()` if needed


---

*This project is part of ongoing research in NLP security and adversarial machine learning.*