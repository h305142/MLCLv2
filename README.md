# AMA-Alignment

Adaptive Multi-scale Affinity Alignment for Hierarchical Contrastive Learning



## Overview

AMA-Alignment is a novel contrastive learning framework that captures both coarse and fine-grained semantic structures through multi-scale region partitioning and adaptive optimization.

**Key Features:**

-  Multi-scale hierarchical region partitioning

-  Adaptive region weight optimization

-  Local contrastive learning with theoretical guarantees

  

## Installation

```

```



## Quick Start

**Train AMA-Alignment:**

```bash
python run.py /path/to/imagenet \
    --use-ama --mlp --aug-plus --cos \
    --multiprocessing-distributed \
    --epochs 200 --batch-size 256
```



```bash
python run.py /path/to/imagenet \
    --mlp --aug-plus --cos \
    --multiprocessing-distributed \
    --epochs 200 --batch-size 256
```

## Configuration

```bash
# Custom AMA parameters
python run.py /path/to/imagenet \
    --use-ama \
    --scales 0.5 0.3 0.1 \
    --num-regions 8 16 32 \
    --rho 1.0
```

## Project Structure

```
ama_alignment/
├── models/           # Model implementations
├── utils/            # Utilities (loss, metrics, optimization)
├── data/             # Data handling and transforms
├── training/         # Training framework
├── config/           # Configuration files
└── run.py            # Main training script
```

