# Data Anonymization Framework

This project implements a comprehensive data anonymization framework with PCA-based transformations, adaptive noise calibration, and clustering algorithms. The framework allows analyzing the impact of anonymization on machine learning models and mathematical properties of datasets.

## Features

- PCA-based anonymization with differential privacy
- Adaptive noise calibration with adjustable parameters
- Stratified clustering for class-aware anonymization
- Feature selection using Chi2 and ExtraTrees methods
- Complete machine learning evaluation pipeline
- Mathematical properties analysis

## Requirements

- Python 3.7+
- NumPy
- SciPy
- scikit-learn
- pandas
- matplotlib (optional for visualization)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/anonymization-framework.git
cd anonymization-framework

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

- `anonimization.py`: Core anonymization algorithms
- `file_utils.py`: Dataset loading and preprocessing utilities
- `ml.py`: Machine learning evaluation functions
- `math_properties.py`: Mathematical property analysis
- `main.py`: Main command-line interface

## Available Datasets

The framework includes support for the following datasets:

1. `adults`: Income classification (label: 'income')
2. `ddos`: DDoS attack detection (label: 'Label')
3. `heart`: Heart disease prediction (label: 'HeartDisease')
4. `cmc`: Contraceptive method choice (label: 'method')
5. `mgm`: MGM dataset (label: 'severity')
6. `cahousing`: California housing (label: 'ocean_proximity')

## Usage

### Basic Usage

Run the default experiment with the California housing dataset:

```bash
python main.py
```

### Specify a Dataset

```bash
python main.py adults
python main.py heart
python main.py mgm
```

### Control Noise Level

Set the noise factor for anonymization (higher values = more noise):

```bash
python main.py mgm --noise=0.01  # Default
python main.py mgm --noise=0.1   # Medium noise
python main.py mgm --noise=0.5   # High noise
python main.py mgm --noise=2.0   # Very high noise
```

### Mathematical Properties Experiment

Run only the mathematical properties analysis:

```bash
python main.py mgm --math-only
```

Run both ML experiment and mathematical properties analysis:

```bash
python main.py mgm --math
```

Specify custom noise levels for mathematical properties experiment:

```bash
python main.py mgm --math-only --noise-levels=0.01,0.1,0.5,2.0
```

### Help Information

Display help and available options:

```bash
python main.py --help
```

## Examples

### Example 1: Run Chi2 and ExtraTree with default noise on Heart dataset

```bash
python main.py heart
```

### Example 2: Analyze mathematical properties of MGM dataset with different noise levels

```bash
python main.py mgm --math-only --noise-levels=0.01,0.05,0.1,0.5,1.0,2.0
```

### Example 3: Run complete analysis on Adults dataset with high noise

```bash
python main.py adults --noise=0.5 --math
```

## Output Files

The experiments produce the following output files in the `results` directory:

- `best_results_chi2_<dataset>_noise_<factor>.csv`: Chi2 experiment results
- `best_results_extra_trees_<dataset>_noise_<factor>.csv`: ExtraTree experiment results
- `math_properties_<dataset>_noise_<factor>.csv`: Individual mathematical properties
- `math_properties_summary_<dataset>.csv`: Summary of mathematical properties across noise levels

## Mathematical Properties Analysis

The framework analyzes the following mathematical properties before and after anonymization:

1. **Basic Statistical Properties**
   - Mean difference
   - Standard deviation difference
   - Skewness difference
   - Kurtosis difference

2. **Structural Properties**
   - Covariance matrix changes
   - Covariance similarity (cosine)

3. **Distance Preservation**
   - Pairwise distance correlation
   - Nearest neighbor preservation

4. **PCA Variance Preservation**
   - Variance preservation ratio
   - Principal component comparison

5. **Class Separation**
   - Class separation correlation

6. **Performance Metrics**
   - Anonymization time

