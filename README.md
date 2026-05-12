<div align="center">

# MAPPO + LSTM Trajectory Prediction

**Multi-Agent Proximal Policy Optimization with LSTM-Based Auxiliary Trajectory Prediction**

<br/>

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/) [![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-EE4C2C?logo=pytorch)](https://pytorch.org/) [![License](https://img.shields.io/badge/License-MIT-green)](LICENSE) [![Conference](https://img.shields.io/badge/AICCC-2025-orange)](https://doi.org/10.1145/3789982.3789990)

<br/>

**[中文文档](README_ZH.md)** | English

> Published at **AICCC 2025** · DOI: [10.1145/3789982.3789990](https://doi.org/10.1145/3789982.3789990) · pp. 46–54

</div>

---

## Overview

Standard MAPPO uses a centralized critic for value estimation but ignores future trajectory information during training. This work augments each agent's actor network with an **LSTM-based trajectory prediction head**, using an auxiliary prediction loss to encourage forward-looking representations.

The key idea: if an agent can accurately predict where its teammates will be in the next few steps, it must have learned a richer, more structured understanding of the environment — which in turn leads to better cooperative policies.

## Key Contributions

| Variant | Description |
|---|---|
| **Baseline MAPPO** | Standard MAPPO with centralized critic, independent actors |
| **MAPPO + Linear** | Actor augmented with a linear trajectory prediction head (`pred_steps=3`) |
| **MAPPO + LSTM v1** | LSTM prediction head with MSE auxiliary loss |
| **MAPPO + LSTM v2** | LSTM prediction head with MSE + InfoNCE contrastive loss |
| **MAPPO + LSTM (full)** | LSTM head + inter-agent multi-head self-attention + InfoNCE loss |

## Architecture

<div align="center">
<img src="arch.png" width="500"/>
</div>

```
Actor (PolicyNet)
├── FC Layers ──────────────────────► Action Distribution  (policy head)
└── LSTM + Multi-head Attention ────► Future State Predictions  (auxiliary head)
        ↑
   Initialized from shared FC embedding

Centralized Critic (CentralValueNet)
└── Concatenated global state [team_size × state_dim] ──► Per-agent value estimates
```

The auxiliary loss is a weighted combination of MSE reconstruction loss and InfoNCE contrastive loss:

```
L_total = L_PPO + α · (L_MSE + β · L_InfoNCE)
```

## Environment

- **Framework**: [ma-gym](https://github.com/koulanurag/ma-gym) Combat
- **Task**: 5v5 team battle on a 20 × 20 grid
- **Observation**: Partial local observation per agent
- **Action space**: Discrete, 5 actions

## Project Structure

```
MAPPO-LSTM-Trajectory/
├── src/
│   ├── mappo/
│   │   └── train.py              # Baseline MAPPO
│   └── mappo_lstm/
│       ├── train_linear.py       # MAPPO + linear prediction head
│       ├── train_lstm_v1.py      # MAPPO + LSTM (MSE loss)
│       ├── train_lstm_v2.py      # MAPPO + LSTM (MSE + InfoNCE)
│       └── train_lstm.py         # MAPPO + LSTM + attention (full model)
├── results/
│   ├── mappo/                    # Baseline training logs & plots
│   └── mappo_lstm/               # LSTM model training logs & plots
├── weights/
│   ├── mappo/                    # Saved baseline weights
│   └── mappo_lstm/               # Saved LSTM model weights
├── requirements.txt
├── README.md                     # This file (English)
└── README_ZH.md                  # 中文文档
```

## Installation

```bash
pip install -r requirements.txt
```

Install ma-gym from source (not on PyPI):

```bash
git clone https://github.com/koulanurag/ma-gym.git
cd ma-gym && pip install -e .
```

## Usage

**Train baseline MAPPO:**
```bash
cd src/mappo
python train.py
```

**Train full MAPPO + LSTM model:**
```bash
cd src/mappo_lstm
python train_lstm.py
```

Training logs are written to `training_log_metrics_weight*.txt`; plots are saved under `plots_metrics_weight/`.

## Hyperparameters

| Parameter | Value |
|---|---|
| Actor LR | 3e-4 |
| Critic LR | 1e-3 |
| Hidden dim | 64 |
| Discount γ | 0.99 |
| GAE λ | 0.97 |
| PPO clip ε | 0.3 |
| Team size | 5 |
| Prediction steps | 5 |
| Auxiliary loss weight α | 0.1 |
| InfoNCE temperature τ | 0.07 |

## Results

Training curves comparing MAPPO baseline vs. MAPPO+LSTM are available in `results/`. The LSTM-augmented model shows improved cumulative reward and faster convergence in the 5v5 combat task.

<div align="center">

| Baseline MAPPO | MAPPO + LSTM (5v5) |
|:---:|:---:|
| <img src="results/mappo_lstm/training_metrics_MAPPO.png" width="380"/> | <img src="results/mappo_lstm/training_metrics_LSTM_5v5.png" width="380"/> |

</div>

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{ji2025mappo,
  title     = {MAPPO with LSTM Trajectory Prediction for Cooperative Multi-Agent Combat},
  author    = {Ji, Jun},
  booktitle = {Proceedings of the 2025 8th International Conference on Algorithms, Computing and Artificial Intelligence (AICCC '25)},
  pages     = {46--54},
  year      = {2025},
  doi       = {10.1145/3789982.3789990},
  publisher = {ACM}
}
```

## License

[MIT](LICENSE)
