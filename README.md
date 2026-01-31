# Disclaimer
**This project is still on my to-do list, and neither the work is finished nor does it reflect my standards of software engineering**


# BipedLoco-RL-CPG

🦿 **Empirical Evaluation of humanoid Biped Locomotion with Central Pattern Generators and Reinforcement Learning**

## 📝 Overview

This project investigates the integration of Central Pattern Generators (CPGs) with Reinforcement Learning algorithms for humanoid bipedal locomotion control. We compare the performance of RL agents with and without CPG integration using the myoSuit package.

**Key Features:**
- CPG-enhanced RL agents for biped locomotion
- Comparative analysis of traditional RL vs. CPG-RL approaches
- Implementation of a hierarchical CPG network based on differential equations as a bioinspired prior in improve sample efficiency
- Comprehensive evaluation metrics using rliable

## 🚀 Quick Start

### Prerequisites
- Python ~3.11
- [uv](https://docs.astral.sh/uv/) (recommended package manager)
- SWIG (for some dependencies)

### Installation

```bash
# Clone the repository
git clone https://github.com/Juxh01/BipedLoco-RL-CPG.git
cd BipedLoco-RL-CPG

# Setup uv
uv venv --python 3.11

# Activate environment (linux)
source .venv/bin/activate

# Install dependencies and setup development environment
make install
```

This will:
- Install SWIG dependency
- Install the project in development mode with all dependencies
- Set up pre-commit hooks for code quality

### Usage

```bash
# Format code
make format

# Run code quality checks
make check

# Run tests
make test

# Build distribution
make build
```
