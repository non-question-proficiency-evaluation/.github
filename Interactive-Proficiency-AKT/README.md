# AKT-Interactive-Proficiency

<div style="font-size:2.5em; font-weight:bold; text-align:center; margin-top:20px;">Context-Aware Attentive Knowledge Tracing</div>

A PyTorch implementation of the Context-Aware Attentive Knowledge Tracing (AKT) model for predicting student performance on educational tasks.

## Overview

This repository implements the AKT model from the paper: [Context-Aware Attentive Knowledge Tracing](https://arxiv.org/abs/2007.12324). The model uses Transformer-based attention mechanisms to track student knowledge states and predict their performance on future questions.

## Features

- **AKT-Rasch Model**: Supports problem difficulty modeling with Problem IDs (PID)
- **AKT-NonRasch Model**: Standard AKT without difficulty parameters
- **Multiple Datasets**: Pre-configured for ASSISTments 2009, 2015, 2017, and Statics datasets
- **5-Fold Cross-Validation**: Built-in support for cross-validation experiments
- **Flexible Architecture**: Configurable transformer blocks, attention heads, and embedding dimensions

## Architecture

The AKT model consists of:

1. **Embedding Layer**: Maps questions and question-answer pairs to dense vectors
2. **Transformer Encoder**: Processes question embeddings with self-attention
3. **Transformer Decoder**: Uses cross-attention between questions and question-answer pairs
4. **Output Layer**: Predicts probability of correct answer

For PID models, additional difficulty parameters are learned for each problem.

## Installation

### Requirements

- Python 3.6+
- PyTorch 1.2.0+
- NumPy 1.17.2+
- Scikit-learn 0.21.3+
- SciPy 1.3.1+

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd Interactive-Proficiency-AKT
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Training

Train AKT-Rasch model on ASSISTments 2009:
```bash
python main.py --dataset assist2009_pid --model akt_pid
```

Train AKT-Rasch model on ASSISTments 2017:
```bash
python main.py --dataset assist2017_pid --model akt_pid
```

Train AKT-NonRasch model on ASSISTments 2015:
```bash
python main.py --dataset assist2015 --model akt_cid
```

### Advanced Options

```bash
python main.py \
    --dataset assist2009_pid \
    --model akt_pid \
    --max_iter 300 \
    --batch_size 24 \
    --lr 1e-5 \
    --d_model 256 \
    --n_block 1 \
    --n_head 8 \
    --dropout 0.05 \
    --train_set 1 \
    --seed 224
```

## Command-Line Arguments

### Basic Parameters
- `--max_iter`: Number of training iterations (default: 300)
- `--train_set`: Fold number for cross-validation (default: 1)
- `--seed`: Random seed for reproducibility (default: 224)

### Training Parameters
- `--batch_size`: Batch size for training (default: 24)
- `--lr`: Learning rate (default: 1e-5)
- `--maxgradnorm`: Maximum gradient norm for clipping (default: -1, disabled)
- `--optim`: Optimizer type (default: 'adam')

### Model Architecture
- `--d_model`: Transformer model dimension (default: 256)
- `--d_ff`: Feed-forward network dimension (default: 1024)
- `--n_block`: Number of transformer blocks (default: 1)
- `--n_head`: Number of attention heads (default: 8)
- `--dropout`: Dropout rate (default: 0.05)
- `--kq_same`: Whether query and key use same weights (default: 1)
- `--final_fc_dim`: Final fully connected layer dimension (default: 512)

### AKT-Rasch Specific
- `--l2`: L2 penalty for difficulty parameters (default: 1e-5)

### Dataset Selection
- `--dataset`: Dataset name (assist2009_pid, assist2017_pid, assist2015, statics)
- `--model`: Model type (akt_pid, akt_cid)

## Datasets

### ASSISTments 2009 (assist2009_pid)
- **Questions**: 110 unique skills
- **Problems**: 16,891 unique problem IDs
- **Format**: 4-line format (Student ID, Problem IDs, Question IDs, Answers)
- **Model**: AKT-Rasch (akt_pid)

### ASSISTments 2015 (assist2015)
- **Questions**: 100 unique skills
- **Format**: 3-line format (Student ID, Question IDs, Answers)
- **Model**: AKT-NonRasch (akt_cid)

### ASSISTments 2017 (assist2017_pid)
- **Questions**: 102 unique skills
- **Problems**: 3,162 unique problem IDs
- **Format**: 4-line format (Student ID, Problem IDs, Question IDs, Answers)
- **Model**: AKT-Rasch (akt_pid)

### Statics (statics)
- **Questions**: 1,223 unique skills
- **Domain**: Engineering/Physics (statics)
- **Format**: 3-line format (Student ID, Question IDs, Answers)

## Data Format

### PID Datasets (assist2009_pid, assist2017_pid)
Each student record consists of 4 lines:
```
<student_id>,<timestamp>
<pid1>,<pid2>,<pid3>,...
<q1>,<q2>,<q3>,...
<answer1>,<answer2>,<answer3>,...
```

### CID Datasets (assist2015, statics)
Each student record consists of 3 lines:
```
<student_id>,<timestamp>
<q1>,<q2>,<q3>,...
<answer1>,<answer2>,<answer3>,...
```

Where:
- `student_id`: Unique student identifier
- `timestamp`: Interaction timestamp
- `pid`: Problem ID (only for PID datasets)
- `q`: Question/Skill ID
- `answer`: Binary answer (0=incorrect, 1=correct)

## Model Outputs

The training process generates:

1. **Model Checkpoints**: Saved in `model/<model_type>/<dataset>/`
   - Best model based on validation AUC
   - Includes model state, optimizer state, and epoch number

2. **Results Logs**: Saved in `result/<model_type>/<dataset>/`
   - Training and validation metrics per epoch
   - AUC, accuracy, and loss values

## Evaluation Metrics

The model reports three key metrics:

- **AUC (Area Under ROC Curve)**: Measures classification performance
- **Accuracy**: Percentage of correct predictions
- **Loss**: Binary cross-entropy loss

## Project Structure

```
Interactive-Proficiency-AKT/
├── akt.py              # AKT model implementation
├── load_data.py        # Data loading utilities
├── run.py              # Training and testing functions
├── utils.py            # Utility functions
├── main.py             # Main training script
├── requirements.txt    # Python dependencies
├── README.md           # This file
├── LICENSE             # License file
└── data/               # Dataset directory
    ├── assist2009_pid/
    ├── assist2015/
    ├── assist2017_pid/
    └── statics/
```

## Hyperparameters

### Recommended Settings

**ASSISTments 2009/2017 (AKT-Rasch)**:
- `d_model`: 256
- `d_ff`: 1024
- `n_block`: 1
- `n_head`: 8
- `dropout`: 0.05
- `batch_size`: 24
- `lr`: 1e-5

**ASSISTments 2015 (AKT-NonRasch)**:
- Same as above, but use `akt_cid` model

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `batch_size`
   - Reduce `d_model` or `d_ff`

2. **Slow Training**
   - Ensure CUDA is available: `torch.cuda.is_available()`
   - Reduce `max_iter` for testing
   - Use smaller dataset subset

3. **Poor Performance**
   - Increase `max_iter`
   - Adjust learning rate
   - Try different `n_block` values

## Citation

If you use this code, please cite the original paper:

```bibtex
@article{akt2020,
  title={Context-Aware Attentive Knowledge Tracing},
  author={Ghosh, Aritra and Heffernan, Neil and Lan, Andrew S},
  journal={arXiv preprint arXiv:2007.12324},
  year={2020}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- Original AKT paper authors
- ASSISTments dataset providers
- PyTorch community
