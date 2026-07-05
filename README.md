<div align="center">
  <h1>Raafeli (CPU Turbo)</h1>
  <p><strong>Zero-config Python decorator to speed up Deep Learning models on CPU by up to 300%.</strong></p>
  
  [![PyPI version](https://badge.fury.io/py/raafeli.svg)](https://badge.fury.io/py/raafeli)
  ![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)
  ![License](https://img.shields.io/badge/License-MIT-green)
</div>

---

## The Problem: GPU-less Deployments
Deploying large AI models or running them on local machines without a dedicated GPU is painfully slow. Matrix multiplications inside `torch.nn.Linear` layers bottleneck heavily on CPU architectures because they process 32-bit floats natively.

## The Solution: Raafeli
**Raafeli** automatically transforms your heavy FP32 PyTorch models into highly optimized INT8 (Dynamic Quantized) representations under the hood. All it takes is a single decorator. You do not need to change your architecture, deployment pipeline, or weights.

---

## Installation

Install Raafeli easily via pip:

```bash
pip install raafeli
```
*(Requires `torch` to be installed in your environment).*

---

## Quick Start

```python
import torch
import torch.nn as nn
from raafeli import optimize_cpu

# 1. Your heavy model
class HeavyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1024, 4096)
        self.fc2 = nn.Linear(4096, 1024)
        
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

model = HeavyModel()
input_data = torch.randn(1, 1024)

# 2. Decorate your prediction function
@optimize_cpu(model_arg="model", precision="int8")
def predict(model, data):
    return model(data)

# 3. Magic! First run takes a tiny fraction of a second to optimize,
#    all subsequent runs execute in INT8 natively on your CPU!
output = predict(model, input_data)
```

## How It Works (Production Features)
When you call `@optimize_cpu`, Raafeli hooks into the execution stack. 

*   **Dynamic Quantization:** It intercepts the `model` object and aggressively applies `torch.quantization.quantize_dynamic` targeting performance-bound layers (like `Linear` and `LSTM`). 
*   **OOP Caching (Zero Overhead):** It caches the optimized model graph directly into the Python object safely. This ensures the overhead is **0ms** on every subsequent call and prevents memory leaks. Your model footprint drops by ~75% and throughput spikes.
*   **Smart Device Guard:** If Raafeli detects that your model is running on a GPU (`.cuda()`), it will gracefully bypass the quantization (since INT8 quantization is exclusively a CPU acceleration technique) without crashing your application.

## Support This Project

Raafeli is an open-source project built out of passion. If it has saved you valuable GPU hours, deployment costs, or debugging time, consider supporting the creator by following on Instagram!

[![Follow on Instagram](https://img.shields.io/badge/Instagram-E4405F?style=for-the-badge&logo=instagram&logoColor=white)](https://instagram.com/galaxy_scream)

---

## Contributing & Testing

We welcome PRs! To run the test suite locally and verify your changes:
```bash
# Clone the repository
git clone https://github.com/ginganomercy/raafeli.git
cd raafeli

# Install with development dependencies
pip install -e .[dev]

# Run tests
pytest tests/
```
