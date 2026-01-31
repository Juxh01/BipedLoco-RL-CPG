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

# Quick Start

### Prerequisites
- Python ~3.11
- [uv](https://docs.astral.sh/uv/) (recommended package manager)
- SWIG (for some dependencies)

### Installation

On a laptop/desktop (with a normal display) you probably want the regular OpenCV wheels
(they include GUI support). Run:

```bash
# Clone the repository
git clone https://github.com/Juxh01/BipedLoco-RL-CPG.git
cd BipedLoco-RL-CPG

# Setup uv virtual env (recommended)
uv venv --python 3.11

# Activate environment (linux)
source .venv/bin/activate

# Install dependencies and setup development environment (desktop OpenCV)
make install
```

On a headless cloud worker (ucloud) you should use the headless OpenCV build to avoid
GUI/X11 dependencies. Use the `ucloud` goal along with `install`:

```bash
make install ucloud
```

Notes:
- `make install` will install the `dev` extra plus the `desktop` extra 
- `make install ucloud` will install the `dev` extra plus the `ucloud` extra 

The install step also installs SWIG and sets up pre-commit hooks.

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
