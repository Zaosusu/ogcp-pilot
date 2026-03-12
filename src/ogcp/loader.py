#!/usr/bin/env python3
"""
OGCP Data Loader

Handles file scanning, JAMS parsing, and PyTorch Dataset integration.

Author: 覃翘 (Qin Qiao) (QQ)
"""

import logging
import warnings
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
import datetime

# Optional PyTorch import
try:
    import torch
    from torch.utils.data import Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Create a dummy Dataset class for type hints
    class Dataset:
        pass

# JAMS library
try:
    import jams
except ImportError:
    raise ImportError(
        "jams library is required. Install with: pip install jams"
    )

from .core import ChordSample

# Configure logging
logger = logging.getLogger(__name__)


def load_jams_annotation(jams_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Parse a JAMS file and extract relevant chord sample metadata.
    
    This function reads a JAMS annotation file and extracts all relevant
    information about the guitar chord sample, including chord name,
    fretboard position, playing technique, and recording metadata.
    
    Args:
        jams_path: Path to the .jams file.
        
    Returns:
        Dict[str, Any]: Dictionary containing extracted metadata:
            - chord_name (str): Standard chord name (e.g., 'C:maj')
            - position (str): Fretboard position
            - technique (str): Playing technique
            - fretboard (list): Fret positions per string
            - duration_sec (float): Audio duration
            - noise_level_db (float, optional): Noise level
            - string_age_days (int, optional): String age
            - recorded_at (datetime, optional): Recording time
            - guitar (str): Guitar model
            - source (str): Recording source
            - sequence (str): Sample sequence number
            
    Raises:
        FileNotFoundError: If the JAMS file does not exist.
        ValueError: If the JAMS file has invalid format.
        
    Example:
        >>> metadata = load_jams_annotation("dataset/raw/C/open-down-001.jams")
        >>> print(metadata['chord_name'], metadata['fretboard'])
    """
    jams_path = Path(jams_path)
    
    if not jams_path.exists():
        raise FileNotFoundError(f"JAMS file not found: {jams_path}")
    
    try:
        jam = jams.load(str(jams_path))
    except Exception as e:
        raise ValueError(f"Failed to parse JAMS file {jams_path}: {e}")
    
    # Extract data from JAMS structure
    result = {}
    
    # Get chord name from first annotation (chord namespace)
    if jam.annotations and len(jam.annotations) > 0:
        chord_annotation = jam.annotations[0]
        if chord_annotation.data and len(chord_annotation.data) > 0:
            result['chord_name'] = chord_annotation.data[0].value
        else:
            result['chord_name'] = 'unknown'
    else:
        result['chord_name'] = 'unknown'
    
    # Get technique from second annotation (tag_open namespace) if available
    result['technique'] = 'unknown'
    if len(jam.annotations) > 1:
        tag_annotation = jam.annotations[1]
        if tag_annotation.namespace == 'tag_open' and tag_annotation.data:
            result['technique'] = tag_annotation.data[0].value
    
    # Extract from file_metadata.identifiers (作为 fallback)
    file_meta = jam.file_metadata
    try:
        identifiers = file_meta.identifiers if hasattr(file_meta, 'identifiers') else {}
        if identifiers is None:
            identifiers = {}
    except Exception:
        identifiers = {}
    
    # Extract duration
    try:
        duration = file_meta.duration
        result['duration_sec'] = float(duration) if duration is not None else 0.0
    except Exception:
        result['duration_sec'] = 0.0
    
    # Extract from sandbox.ogcp (优先使用，数据更完整)
    try:
        sandbox = jam.sandbox
        if sandbox is not None and hasattr(sandbox, 'ogcp'):
            ogcp_data = sandbox.ogcp
            if ogcp_data is None:
                ogcp_data = {}
        else:
            ogcp_data = {}
    except Exception:
        ogcp_data = {}
    
    # 确保 ogcp_data 是字典
    if not isinstance(ogcp_data, dict):
        ogcp_data = {}
    
    # 优先使用 ogcp 数据，如果没有再用 identifiers，最后使用默认值
    # Position: ogcp -> identifiers -> 'unknown'
    if 'position' in ogcp_data and ogcp_data['position']:
        result['position'] = ogcp_data['position']
    elif isinstance(identifiers, dict) and identifiers.get('position'):
        result['position'] = identifiers['position']
    else:
        result['position'] = 'unknown'
    
    # Guitar: ogcp -> identifiers -> 'unknown'
    if 'guitar' in ogcp_data and ogcp_data['guitar']:
        result['guitar'] = ogcp_data['guitar']
    elif isinstance(identifiers, dict) and identifiers.get('guitar'):
        result['guitar'] = identifiers['guitar']
    else:
        result['guitar'] = 'unknown'
    
    # Source: ogcp -> identifiers -> 'unknown'
    if 'source' in ogcp_data and ogcp_data['source']:
        result['source'] = ogcp_data['source']
    elif isinstance(identifiers, dict) and identifiers.get('source'):
        result['source'] = identifiers['source']
    else:
        result['source'] = 'unknown'
    
    # Sequence: ogcp -> identifiers -> '000'
    if 'sequence' in ogcp_data and ogcp_data['sequence']:
        result['sequence'] = ogcp_data['sequence']
    elif isinstance(identifiers, dict) and identifiers.get('sequence'):
        result['sequence'] = identifiers['sequence']
    else:
        result['sequence'] = '000'
    
    # Technique: ogcp 可以覆盖 annotation 的值（ogcp 更准确）
    if 'technique' in ogcp_data and ogcp_data['technique']:
        result['technique'] = ogcp_data['technique']
    
    # 从 ogcp 获取其他数据
    result['fretboard'] = ogcp_data.get('fretboard', ['x', 'x', 'x', 'x', 'x', 'x'])
    result['noise_level_db'] = ogcp_data.get('noise_level_db')
    result['string_age_days'] = ogcp_data.get('string_age_days')
    
    # Parse recording timestamp
    recorded_at_str = ogcp_data.get('recorded_at')
    if recorded_at_str:
        try:
            result['recorded_at'] = datetime.datetime.fromisoformat(recorded_at_str)
        except ValueError:
            result['recorded_at'] = None
    else:
        result['recorded_at'] = None
    
    return result


class OGCPDataset(Dataset):
    """
    PyTorch Dataset for OpenGuitarChordProject data.
    
    This dataset recursively scans the OGCP directory structure,
    automatically pairs WAV files with their JAMS annotations,
    and provides access to ChordSample objects.
    
    Attributes:
        root_dir (Path): Root directory containing the dataset.
        samples (List[ChordSample]): List of loaded samples.
        transform (callable, optional): Optional transform function.
        
    Args:
        root_dir: Path to dataset root (e.g., "dataset/raw").
        transform: Optional transform to apply to samples.
        chord_filter: Optional list of chord names to include.
        position_filter: Optional list of positions to include.
        
    Example:
        >>> dataset = OGCPDataset(root_dir="dataset/raw")
        >>> print(len(dataset))
        660
        >>> sample = dataset[0]
        >>> print(sample.chord_name, sample.fretboard)
        >>> 
        >>> # Use with PyTorch DataLoader
        >>> from torch.utils.data import DataLoader
        >>> loader = DataLoader(dataset, batch_size=32, shuffle=True)
    """
    
    def __init__(
        self,
        root_dir: Union[str, Path],
        transform: Optional[callable] = None,
        chord_filter: Optional[List[str]] = None,
        position_filter: Optional[List[str]] = None,
        technique_filter: Optional[List[str]] = None,
    ):
        """
        Initialize the OGCP Dataset.
        
        Args:
            root_dir: Root directory of the dataset.
            transform: Optional transform function.
            chord_filter: List of chord names to filter (e.g., ['C:maj', 'G:maj']).
            position_filter: List of positions to filter (e.g., ['open', 'barre3']).
            technique_filter: List of techniques to filter (e.g., ['down', 'arp']).
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for OGCPDataset. "
                "Install with: pip install torch"
            )
        
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.chord_filter = chord_filter
        self.position_filter = position_filter
        self.technique_filter = technique_filter
        
        self.samples: List[ChordSample] = []
        self._scan_and_load()
    
    def _scan_and_load(self) -> None:
        """
        Recursively scan directory and load all valid samples.
        
        This method walks through the dataset directory structure,
        finds all WAV files, attempts to pair them with JAMS annotations,
        and creates ChordSample objects. Missing JAMS files trigger warnings.
        """
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.root_dir}")
        
        wav_files = list(self.root_dir.rglob("*.wav"))
        logger.info(f"Found {len(wav_files)} WAV files in {self.root_dir}")
        
        skipped_count = 0
        
        for wav_path in wav_files:
            jams_path = wav_path.with_suffix('.jams')
            
            # Check if JAMS annotation exists
            if not jams_path.exists():
                warnings.warn(
                    f"Missing JAMS annotation for {wav_path.name}, skipping. "
                    f"Expected: {jams_path.name}"
                )
                skipped_count += 1
                continue
            
            try:
                # Load JAMS metadata
                metadata = load_jams_annotation(jams_path)
                
                # Apply filters
                if self.chord_filter and metadata['chord_name'] not in self.chord_filter:
                    continue
                if self.position_filter and metadata['position'] not in self.position_filter:
                    continue
                if self.technique_filter and metadata['technique'] not in self.technique_filter:
                    continue
                
                # Create ChordSample
                sample = ChordSample(
                    audio_path=wav_path,
                    chord_name=metadata['chord_name'],
                    position=metadata['position'],
                    technique=metadata['technique'],
                    fretboard=metadata['fretboard'],
                    duration_sec=metadata['duration_sec'],
                    noise_level_db=metadata.get('noise_level_db'),
                    string_age_days=metadata.get('string_age_days'),
                    recorded_at=metadata.get('recorded_at'),
                    guitar=metadata.get('guitar', 'unknown'),
                    source=metadata.get('source', 'unknown'),
                    sequence=metadata.get('sequence', '000'),
                )
                
                self.samples.append(sample)
                
            except Exception as e:
                logger.warning(f"Failed to load sample {wav_path}: {e}")
                skipped_count += 1
                continue
        
        logger.info(
            f"Successfully loaded {len(self.samples)} samples, "
            f"skipped {skipped_count} files"
        )
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> ChordSample:
        """
        Get a sample by index.
        
        Args:
            idx: Sample index.
            
        Returns:
            ChordSample: The chord sample at the given index.
            
        Note:
            If a transform is set, it will be applied to the sample.
        """
        sample = self.samples[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample
    
    def get_by_chord(self, chord_name: str) -> List[ChordSample]:
        """
        Get all samples for a specific chord.
        
        Args:
            chord_name: Chord name (e.g., 'C:maj', 'A:min').
            
        Returns:
            List[ChordSample]: All samples matching the chord.
        """
        return [s for s in self.samples if s.chord_name == chord_name]
    
    def get_chord_distribution(self) -> Dict[str, int]:
        """
        Get distribution of chords in the dataset.
        
        Returns:
            Dict[str, int]: Mapping from chord name to count.
        """
        distribution = {}
        for sample in self.samples:
            distribution[sample.chord_name] = distribution.get(sample.chord_name, 0) + 1
        return dict(sorted(distribution.items()))
    
    def statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive dataset statistics.
        
        Returns:
            Dict containing:
                - total_samples: Total number of samples
                - unique_chords: Number of unique chords
                - unique_positions: Number of unique positions
                - unique_techniques: Number of unique techniques
                - chord_distribution: Dict of chord counts
                - avg_duration: Average audio duration
        """
        if not self.samples:
            return {
                'total_samples': 0,
                'unique_chords': 0,
                'unique_positions': 0,
                'unique_techniques': 0,
                'chord_distribution': {},
                'avg_duration_sec': 0.0,
            }
        
        chords = set()
        positions = set()
        techniques = set()
        total_duration = 0.0
        
        for sample in self.samples:
            chords.add(sample.chord_name)
            positions.add(sample.position)
            techniques.add(sample.technique)
            total_duration += sample.duration_sec
        
        return {
            'total_samples': len(self.samples),
            'unique_chords': len(chords),
            'unique_positions': len(positions),
            'unique_techniques': len(techniques),
            'chord_distribution': self.get_chord_distribution(),
            'avg_duration_sec': total_duration / len(self.samples),
        }
