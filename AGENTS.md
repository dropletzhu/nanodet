# AGENTS.md - Guidelines for Agentic Coding in NanoDet

## Project Overview

NanoDet is a lightweight anchor-free object detection model built with PyTorch and PyTorch Lightning. This file provides guidelines for agents working on this codebase.

---

## Build, Lint, and Test Commands

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Setup package in development mode
python setup.py develop
```

### Linting

```bash
# Run isort (import sorting)
isort --profile black .

# Run black (code formatting)
black .

# Run flake8 (linting)
flake8 .
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run tests with coverage
coverage run --branch --source nanodet -m pytest tests/
coverage report -m

# Run a single test file
pytest tests/test_models/test_backbone/test_shufflenetv2.py

# Run a single test function
pytest tests/test_models/test_backbone/test_shufflenetv2.py::test_shufflenetv2

# Run tests in parallel (requires pytest-xdist)
pytest -n auto tests/
```

### Training

```bash
# Train model
python tools/train.py CONFIG_FILE_PATH
```

---

## Code Style Guidelines

### Formatting

- **Max line length**: 88 characters (Black default)
- **Line ending**: LF
- **Indentation**: 4 spaces
- **Quote style**: Double quotes (`"`)

### Import Organization (isort)

Imports must be sorted with `isort --profile black`. The order:

1. Standard library imports
2. Third-party imports
3. Local/relative imports (`from nanodet...`)

Example:
```python
import os
import sys

import torch
import torch.nn as nn

from nanodet.model.backbone import build_backbone
```

### Naming Conventions

| Element | Convention | Example |
|---------|------------|---------|
| Classes | PascalCase | `ShuffleNetV2`, `NanoDetHead` |
| Functions/methods | snake_case | `channel_shuffle()`, `forward()` |
| Variables | snake_case | `input_channels`, `output` |
| Constants | UPPER_SNAKE | `MAX_BOXES`, `NUM_CLASSES` |
| Private methods | _snake_case | `_initialize_weights()` |

### Type Hints

Use type hints where beneficial. PyTorch/numpy types are allowed:

```python
def channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
    """Shuffle channels across groups.
    
    Args:
        x: Input tensor of shape (B, C, H, W)
        groups: Number of groups to split channels into
        
    Returns:
        Shuffled tensor with same shape as input
    """
    batchsize, num_channels, height, width = x.data.size()
    ...
```

### Docstrings

- Use triple double quotes `"""` for docstrings
- Follow NumPy style for complex functions:
```python
def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Forward pass of the network.
    
    Args:
        x: Input tensor of shape (B, C, H, W)
        
    Returns:
        Tuple of output tensors at different scales
    """
```

### Error Handling

- Use specific exceptions (`ValueError`, `NotImplementedError`, `AssertionError`)
- Validate inputs at function entry:
```python
if not (1 <= stride <= 3):
    raise ValueError("illegal stride value")
```

### PyTorch Conventions

- Use `nn.Module` base class for all neural network modules
- Initialize weights in `_initialize_weights()` method
- Use `super().__init__()` in constructors
- Store configuration as instance attributes
- Use `nn.Sequential` for simple stacks of layers

### Testing Guidelines

- Test files go in `tests/` mirror the source structure
- Use `pytest` framework
- Test names start with `test_`
- Use `torch.rand()` for random inputs
- Test both success and failure cases:
```python
def test_shufflenetv2():
    with pytest.raises(NotImplementedError):
        build_backbone(cfg)  # Test invalid config raises error
    
    model = ShuffleNetV2(...)
    output = model(input)
    assert output[0].shape == (1, 48, 8, 8)
```

### Pre-commit Hooks

This project uses pre-commit. Install and run:

```bash
pip install pre-commit
pre-commit install
```

This will automatically run isort, black, and flake8 on staged files.

---

## Project Structure

```
nanodet/
├── nanodet/           # Main source code
│   ├── model/         # Neural network models
│   │   ├── backbone/  # Backbone networks
│   │   ├── fpn/      # Feature pyramid networks
│   │   ├── head/     # Detection heads
│   │   └── loss/     # Loss functions
│   ├── data/         # Data loading and augmentation
│   ├── evaluator/    # Evaluation metrics
│   ├── trainer/      # Training logic (PyTorch Lightning)
│   └── util/         # Utilities
├── tests/             # Test suite (mirrors nanodet/ structure)
├── config/            # Configuration files
├── demo/              # Demo scripts
└── tools/             # Training/inference tools
```

---

## Key Dependencies

- torch >= 1.10, < 2.0
- pytorch-lightning >= 1.9.0, < 2.0.0
- pyyaml (config parsing)
- pytest (testing)
- black, isort, flake8 (linting)