#!/usr/bin/env python3
"""
OpenGuitarChordProject (OGCP) Python SDK

A professional Python SDK for accessing the OpenGuitarChordProject dataset,
providing standardized guitar chord samples with JAMS annotations.

Author: 覃翘 (Qin Qiao) (QQ)
Email: qinqiao2014@gmail.com
Version: 1.0.0
License: MIT

Example:
    >>> from ogcp import OGCPDataset, ChordSample
    >>> dataset = OGCPDataset(root_dir="dataset/raw")
    >>> sample = dataset[0]
    >>> print(sample.chord_name, sample.fretboard)
"""

__version__ = "1.0.0"
__author__ = "覃翘 (Qin Qiao) (QQ)"
__email__ = "qinqiao2014@gmail.com"

from .core import ChordSample
from .loader import OGCPDataset, load_jams_annotation
from .viz import plot_fretboard, plot_chord_spectrogram

__all__ = [
    "ChordSample",
    "OGCPDataset", 
    "load_jams_annotation",
    "plot_fretboard",
    "plot_chord_spectrogram",
]
