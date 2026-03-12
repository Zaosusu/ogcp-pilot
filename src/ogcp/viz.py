#!/usr/bin/env python3
"""
OGCP Visualization Module

Provides visualization tools for guitar chord fingerboard and audio analysis.

Author: 覃翘 (Qin Qiao) (QQ)
"""

import logging
from pathlib import Path
from typing import List, Union, Optional, Tuple
import warnings

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle
import numpy as np

# Optional librosa for audio visualization
try:
    import librosa
    import librosa.display
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

from .core import ChordSample

logger = logging.getLogger(__name__)

# Guitar string names (standard tuning)
STRING_NAMES = ['E', 'A', 'D', 'G', 'B', 'e']  # 6th to 1st string


def plot_fretboard(
    fretboard: List[Union[int, str]],
    chord_name: Optional[str] = None,
    position: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 4),
    show_fret_numbers: bool = True,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Plot a guitar fingerboard diagram with fret positions.
    
    This function creates a visual representation of a guitar chord
    showing which frets are pressed on each string. Open strings are
    marked with 'O', muted strings with 'X'.
    
    Args:
        fretboard: List of fret positions for strings [6,5,4,3,2,1].
            Use 'x' for muted strings, integers for fret numbers.
        chord_name: Optional chord name to display in title.
        position: Optional position name (e.g., 'open', 'barre3').
        figsize: Figure size as (width, height).
        show_fret_numbers: Whether to show fret number labels.
        save_path: Optional path to save the figure.
        
    Returns:
        plt.Figure: The matplotlib figure object.
        
    Raises:
        ValueError: If fretboard has wrong number of elements.
        
    Example:
        >>> from ogcp import plot_fretboard
        >>> # C major open chord: x32010
        >>> fig = plot_fretboard(
        ...     fretboard=['x', 3, 2, 0, 1, 0],
        ...     chord_name='C:maj',
        ...     position='open'
        ... )
        >>> plt.show()
    """
    if len(fretboard) != 6:
        raise ValueError(f"Fretboard must have 6 elements, got {len(fretboard)}")
    
    # Determine fret range to display
    frets = [f for f in fretboard if isinstance(f, int) and f > 0]
    if frets:
        max_fret = max(frets)
        min_display_fret = min(frets) if max_fret <= 5 else max(1, min(frets) - 1)
        max_display_fret = max(max_fret + 1, 5)
    else:
        # All open or muted
        min_display_fret = 0
        max_display_fret = 5
    
    num_frets = max_display_fret - min_display_fret + 1
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Draw fretboard background
    fretboard_width = 6  # 6 strings
    fretboard_height = num_frets - 1  # fret spacing
    
    # Draw strings (vertical lines)
    for i in range(6):
        x = i
        ax.axvline(x=x, color='#8B4513', linewidth=2, zorder=1)
    
    # Draw frets (horizontal lines)
    for i in range(num_frets):
        y = i
        ax.axhline(y=y, color='#696969', linewidth=1.5, zorder=1)
        # Make fret wire thicker
        if i > 0:
            ax.axhline(y=y, color='#C0C0C0', linewidth=3, zorder=1)
    
    # Draw nut (thick line at top for open positions)
    if min_display_fret == 0:
        ax.axhline(y=0, color='#8B4513', linewidth=8, zorder=2)
    
    # Plot finger positions
    for string_idx, fret in enumerate(fretboard):
        x = string_idx
        string_name = STRING_NAMES[string_idx]
        
        if fret == 'x' or fret == 'X':
            # Muted string - draw X at top
            ax.text(x, num_frets - 0.3, '×', fontsize=20, 
                   ha='center', va='center', color='red', fontweight='bold')
        elif fret == 0:
            # Open string - draw O at top
            circle = Circle((x, num_frets - 0.3), 0.15, 
                          facecolor='none', edgecolor='green', linewidth=2)
            ax.add_patch(circle)
        else:
            # Fret position
            if min_display_fret <= fret <= max_display_fret:
                y = fret - min_display_fret
                # Draw finger dot
                circle = Circle((x, y), 0.25, 
                              facecolor='#4169E1', edgecolor='black', linewidth=1)
                ax.add_patch(circle)
                # Add fret number inside circle
                ax.text(x, y, str(fret), fontsize=10, 
                       ha='center', va='center', color='white', fontweight='bold')
    
    # Add string labels at bottom
    for i, name in enumerate(STRING_NAMES):
        ax.text(i, -0.5, name, fontsize=12, ha='center', va='center', fontweight='bold')
    
    # Add fret numbers on left side
    if show_fret_numbers:
        for i in range(num_frets):
            fret_num = min_display_fret + i
            if fret_num > 0:
                ax.text(-0.7, i, str(fret_num), fontsize=10, 
                       ha='center', va='center')
    
    # Set axis properties
    ax.set_xlim(-1, 6)
    ax.set_ylim(-1, num_frets + 0.5)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Add title
    title = "Guitar Fingerboard"
    if chord_name:
        title = f"{chord_name}"
        if position:
            title += f" ({position})"
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
                  markersize=10, label='Open String'),
        plt.Line2D([0], [0], marker='x', color='w', markerfacecolor='red', 
                  markeredgecolor='red', markersize=10, label='Muted'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#4169E1', 
                  markersize=10, label='Fret Position'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', 
             bbox_to_anchor=(1.15, 1), frameon=True)
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        logger.info(f"Saved fingerboard plot to {save_path}")
    
    return fig


def plot_chord_spectrogram(
    sample: ChordSample,
    figsize: Tuple[int, int] = (12, 4),
    save_path: Optional[Union[str, Path]] = None,
) -> Optional[plt.Figure]:
    """
    Plot a spectrogram of a chord sample.
    
    This function loads the audio file from a ChordSample and displays
    its spectrogram using librosa.
    
    Args:
        sample: ChordSample containing audio_path.
        figsize: Figure size as (width, height).
        save_path: Optional path to save the figure.
        
    Returns:
        plt.Figure or None: The matplotlib figure object, or None if librosa
            is not available.
            
    Raises:
        FileNotFoundError: If audio file does not exist.
        
    Example:
        >>> from ogcp import OGCPDataset, plot_chord_spectrogram
        >>> dataset = OGCPDataset("dataset/raw")
        >>> sample = dataset[0]
        >>> fig = plot_chord_spectrogram(sample)
        >>> plt.show()
    """
    if not LIBROSA_AVAILABLE:
        warnings.warn("librosa not available. Install with: pip install librosa")
        return None
    
    if not sample.audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {sample.audio_path}")
    
    # Load audio
    try:
        y, sr = librosa.load(str(sample.audio_path), sr=None)
    except Exception as e:
        logger.error(f"Failed to load audio {sample.audio_path}: {e}")
        raise
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot waveform
    ax1 = axes[0]
    librosa.display.waveshow(y, sr=sr, ax=ax1, color='#4169E1')
    ax1.set_title(f"Waveform: {sample.chord_name}")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.grid(True, alpha=0.3)
    
    # Plot spectrogram
    ax2 = axes[1]
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', 
                                    ax=ax2, cmap='viridis')
    ax2.set_title(f"Spectrogram: {sample.chord_name}")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Frequency (Hz)")
    fig.colorbar(img, ax=ax2, format='%+2.0f dB')
    
    plt.tight_layout()
    plt.suptitle(f"{sample.chord_name} - {sample.position}/{sample.technique}", 
                fontsize=14, y=1.02)
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved spectrogram to {save_path}")
    
    return fig


def compare_chords(
    samples: List[ChordSample],
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Compare multiple chord fingerboards side by side.
    
    Args:
        samples: List of ChordSample objects to compare.
        figsize: Figure size.
        save_path: Optional path to save the figure.
        
    Returns:
        plt.Figure: The matplotlib figure object.
        
    Example:
        >>> from ogcp import OGCPDataset, compare_chords
        >>> dataset = OGCPDataset("dataset/raw")
        >>> c_samples = dataset.get_by_chord("C:maj")[:3]
        >>> fig = compare_chords(c_samples)
        >>> plt.show()
    """
    n_samples = len(samples)
    if n_samples == 0:
        raise ValueError("No samples provided for comparison")
    
    # Calculate subplot layout
    n_cols = min(4, n_samples)
    n_rows = (n_samples + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_samples == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_rows > 1 else axes
    
    for idx, (ax, sample) in enumerate(zip(axes, samples)):
        # Draw simplified fretboard
        _draw_mini_fretboard(ax, sample.fretboard)
        
        # Set title
        title = f"{sample.chord_name}\n{sample.position}/{sample.technique}"
        ax.set_title(title, fontsize=10)
        ax.axis('off')
    
    # Hide unused subplots
    for idx in range(n_samples, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved comparison to {save_path}")
    
    return fig


def _draw_mini_fretboard(ax, fretboard: List[Union[int, str]]) -> None:
    """
    Helper function to draw a mini fretboard on a given axis.
    
    Args:
        ax: Matplotlib axis to draw on.
        fretboard: Fret positions for 6 strings.
    """
    # Simple 4-fret display
    num_frets = 4
    
    # Draw grid
    for i in range(7):  # 6 strings + edge
        x = i - 0.5
        ax.axvline(x=x, color='#8B4513', linewidth=1, alpha=0.7)
    
    for i in range(num_frets + 1):
        y = i - 0.5
        ax.axhline(y=y, color='#696969', linewidth=1, alpha=0.7)
    
    # Draw positions
    for string_idx, fret in enumerate(fretboard):
        x = string_idx
        
        if fret == 'x' or fret == 'X':
            ax.text(x, num_frets - 0.8, '×', fontsize=16, 
                   ha='center', va='center', color='red')
        elif fret == 0:
            circle = Circle((x, num_frets - 0.8), 0.2, 
                          facecolor='none', edgecolor='green', linewidth=1.5)
            ax.add_patch(circle)
        elif isinstance(fret, int) and 1 <= fret <= 4:
            y = num_frets - fret - 0.5
            circle = Circle((x, y), 0.25, facecolor='#4169E1', edgecolor='black')
            ax.add_patch(circle)
            ax.text(x, y, str(fret), fontsize=8, ha='center', va='center', 
                   color='white', fontweight='bold')
    
    ax.set_xlim(-0.7, 5.7)
    ax.set_ylim(-0.7, 3.7)
    ax.set_aspect('equal')
