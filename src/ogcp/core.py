#!/usr/bin/env python3
"""
OGCP Core Data Models

Defines the core data structures for representing guitar chord samples.

Author: 覃翘 (Qin Qiao) (QQ)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union
import datetime


@dataclass
class ChordSample:
    """
    Represents a single guitar chord sample with all metadata.
    
    This class encapsulates all dimensions of a chord sample including
    audio path, chord information, playing technique, fretboard position,
    and recording metadata.
    
    Attributes:
        audio_path (Path): Path to the WAV audio file.
        chord_name (str): Standard chord name (e.g., 'C:maj', 'A:min').
        position (str): Fretboard position (e.g., 'open', 'barre3').
        technique (str): Playing technique (e.g., 'down', 'arp', 'mute').
        fretboard (List[Union[int, str]]): Fret positions for strings [6,5,4,3,2,1].
            Use 'x' for muted strings, integers for fret numbers.
        duration_sec (float): Audio duration in seconds.
        noise_level_db (Optional[float]): Background noise level in dB.
        string_age_days (Optional[int]): Days since last string change.
        recorded_at (Optional[datetime.datetime]): Recording timestamp.
        guitar (str): Guitar model used for recording.
        source (str): Recording source/method.
        
    Example:
        >>> sample = ChordSample(
        ...     audio_path=Path("dataset/raw/C/open-down-enya-direct-001.wav"),
        ...     chord_name="C:maj",
        ...     position="open",
        ...     technique="down",
        ...     fretboard=['x', 3, 2, 0, 1, 0],
        ...     duration_sec=4.52,
        ...     noise_level_db=-45.2,
        ...     string_age_days=434,
        ...     guitar="NEXG2xCCS",
        ...     source="direct"
        ... )
    """
    
    # Required fields
    audio_path: Path
    chord_name: str
    position: str
    technique: str
    fretboard: List[Union[int, str]]
    duration_sec: float
    
    # Optional metadata fields
    noise_level_db: Optional[float] = None
    string_age_days: Optional[int] = None
    recorded_at: Optional[datetime.datetime] = None
    guitar: str = "unknown"
    source: str = "unknown"
    sequence: str = "000"
    
    def __post_init__(self):
        """Validate fretboard coordinates after initialization."""
        if len(self.fretboard) != 6:
            raise ValueError(
                f"Fretboard must have exactly 6 elements (one per string), "
                f"got {len(self.fretboard)}"
            )
    
    @property
    def jamspath(self) -> Path:
        """
        Get the corresponding JAMS annotation file path.
        
        Returns:
            Path: Path to the .jams file with same base name.
        """
        return self.audio_path.with_suffix('.jams')
    
    @property
    def chord_root(self) -> str:
        """
        Extract root note from chord name.
        
        Returns:
            str: Root note (e.g., 'C', 'F#', 'Bb').
        """
        return self.chord_name.split(':')[0] if ':' in self.chord_name else self.chord_name
    
    @property
    def chord_quality(self) -> str:
        """
        Extract chord quality from chord name.
        
        Returns:
            str: Chord quality (e.g., 'maj', 'min', 'dim').
        """
        parts = self.chord_name.split(':')
        return parts[1] if len(parts) > 1 else 'maj'
    
    def is_barre(self) -> bool:
        """
        Check if this is a barre chord position.
        
        Returns:
            bool: True if position starts with 'barre'.
        """
        return self.position.startswith('barre')
    
    def get_string_fret(self, string_num: int) -> Union[int, str]:
        """
        Get fret position for a specific string.
        
        Args:
            string_num: String number (1-6, where 1 is high E).
            
        Returns:
            Union[int, str]: Fret number or 'x' for muted.
            
        Raises:
            ValueError: If string_num is not in range 1-6.
        """
        if not 1 <= string_num <= 6:
            raise ValueError(f"String number must be 1-6, got {string_num}")
        # string_num 1 = index 5, string_num 6 = index 0
        return self.fretboard[6 - string_num]
    
    def __repr__(self) -> str:
        """String representation of the sample."""
        return (
            f"ChordSample({self.chord_name}, "
            f"{self.position}/{self.technique}, "
            f"fretboard={self.fretboard})"
        )
    
    def to_dict(self) -> dict:
        """
        Convert sample to dictionary representation.
        
        Returns:
            dict: Dictionary containing all sample attributes.
        """
        return {
            'audio_path': str(self.audio_path),
            'chord_name': self.chord_name,
            'position': self.position,
            'technique': self.technique,
            'fretboard': self.fretboard,
            'duration_sec': self.duration_sec,
            'noise_level_db': self.noise_level_db,
            'string_age_days': self.string_age_days,
            'recorded_at': self.recorded_at.isoformat() if self.recorded_at else None,
            'guitar': self.guitar,
            'source': self.source,
            'sequence': self.sequence,
        }
