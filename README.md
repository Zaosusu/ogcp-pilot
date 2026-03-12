# OGCP-Pilot: A Physics-Aware Guitar Chord Dataset for Robust Recognition

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Hugging Face Datasets](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue)](https://huggingface.co/datasets/Zaosusu/ogcp-pilot)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18979053.svg)](https://doi.org/10.5281/zenodo.18979053)

> **English** | [中文介绍](#中文介绍)

**OGCP (Open Guitar Chord Project) Pilot Study v1.0**

660 high-fidelity guitar chord samples with physical annotations for domain-robust chord recognition.

## Quick Start

```python
from ogcp import OGCPDataset

dataset = OGCPDataset(root_dir='dataset/raw')
print(len(dataset))  # 660
```

> For training and inference, see [`ml/README.md`](ml/README.md).

## Data Distribution

| Component | Location |
|:---|:---|
| **Code & SDK** | This GitHub repo |
| **Annotations** | `dataset/raw/*.jams` |
| **Audio Files** | [Hugging Face](https://huggingface.co/datasets/Zaosusu/ogcp-pilot) |

## Dataset Statistics

| Attribute | Value |
|:---|:---|
| Total Samples | 660 |
| Chord Classes | 14 |
| Recording Device | Enya NEXG2xCCS |

---

## 中文介绍

**OGCP (Open Guitar Chord Project) Pilot Study v1.0**

660 个高保真吉他和弦样本，包含物理标注，用于鲁棒性和弦识别研究。

### 快速开始

```python
from ogcp import OGCPDataset

dataset = OGCPDataset(root_dir='dataset/raw')
print(len(dataset))  # 660
```

> 训练与推理说明请见 [`ml/README.md`](ml/README.md)。

### 数据分布

| 组件 | 位置 |
|:---|:---|
| **代码与 SDK** | 本 GitHub 仓库 |
| **标注文件** | `dataset/raw/*.jams` |
| **音频文件** | [Hugging Face](https://huggingface.co/datasets/Zaosusu/ogcp-pilot) |

### 数据集统计

| 属性 | 数值 |
|:---|:---|
| 总样本数 | 660 |
| 和弦类别 | 14 |
| 录音设备 | Enya NEXG2xCCS |

---

## Citation / 引用

```bibtex
@dataset{qiao2026ogcp,
  author = {Qin Qiao},
  title = {OGCP Pilot: A Physics-Aware Guitar Chord Dataset},
  year = {2026},
  version = {1.0},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.18979053},
  url = {https://doi.org/10.5281/zenodo.18979053}
}
```
