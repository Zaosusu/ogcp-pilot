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

HF_REPO_ID = "Zaosusu/ogcp-pilot"


def _download_from_hf(root_dir: Path) -> None:
    """
    Download audio files from Hugging Face if local directory is empty.

    Args:
        root_dir: Local directory to download files into.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required for auto-download. "
            "Install with: pip install huggingface_hub"
        )

    print(f"Local dataset not found. Downloading from Hugging Face ({HF_REPO_ID})...")
    snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        local_dir=str(root_dir),
    )
    print(f"Download complete. Files saved to: {root_dir}")


def load_jams_annotation(jams_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Parse a JAMS file and extract relevant chord sample metadata.
    """
    jams_path = Path(jams_path)
    
    if not jams_path.exists():
        raise FileNotFoundError(f"JAMS file not found: {jams_path}")
    
    try:
        jam = jams.load(str(jams_path))
    except Exception as e:
        raise ValueError(f"Failed to parse JAMS file {jams_path}: {e}")
    
    result = {}
    
    # Get chord name from first annotation
    if jam.annotations and len(jam.annotations) > 0:
        chord_annotation = jam.annotations[0]
        if chord_annotation.data and len(chord_annotation.data) > 0:
            result['chord_name'] = chord_annotation.data[0].value
        else:
            result['chord_name'] = 'unknown'
    else:
        result['chord_name'] = 'unknown'
    
    # Get technique from second annotation
    result['technique'] = 'unknown'
    if len(jam.annotations) > 1:
        tag_annotation = jam.annotations[1]
        if tag_annotation.namespace == 'tag_open' and tag_annotation.data:
            result['technique'] = tag_annotation.data[0].value
    
    # Extract from file_metadata
    file_meta = jam.file_metadata
    try:
        identifiers = file_meta.identifiers if hasattr(file_meta, 'identifiers') else {}
        if identifiers is None:
            identifiers = {}
    except Exception:
        identifiers = {}
    
    try:
        duration = file_meta.duration
        result['duration_sec'] = float(duration) if duration is not None else 0.0
    except Exception:
        result['duration_sec'] = 0.0
    
    # Extract from sandbox.ogcp
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
    
    if not isinstance(ogcp_data, dict):
        ogcp_data = {}
    
    # Priority: ogcp -> identifiers -> default
    if 'position' in ogcp_data and ogcp_data['position']:
        result['position'] = ogcp_data['position']
    elif isinstance(identifiers, dict) and identifiers.get('position'):
        result['position'] = identifiers['position']
    else:
        result['position'] = 'unknown'
    
    if 'guitar' in ogcp_data and ogcp_data['guitar']:
        result['guitar'] = ogcp_data['guitar']
    elif isinstance(identifiers, dict) and identifiers.get('guitar'):
        result['guitar'] = identifiers['guitar']
    else:
        result['guitar'] = 'unknown'
    
    if 'source' in ogcp_data and ogcp_data['source']:
        result['source'] = ogcp_data['source']
    elif isinstance(identifiers, dict) and identifiers.get('source'):
        result['source'] = identifiers['source']
    else:
        result['source'] = 'unknown'
    
    if 'sequence' in ogcp_data and ogcp_data['sequence']:
        result['sequence'] = ogcp_data['sequence']
    elif isinstance(identifiers, dict) and identifiers.get('sequence'):
        result['sequence'] = identifiers['sequence']
    else:
        result['sequence'] = '000'
    
    if 'technique' in ogcp_data and ogcp_data['technique']:
        result['technique'] = ogcp_data['technique']
    
    result['fretboard'] = ogcp_data.get('fretboard', ['x', 'x', 'x', 'x', 'x', 'x'])
    result['noise_level_db'] = ogcp_data.get('noise_level_db')
    result['string_age_days'] = ogcp_data.get('string_age_days')
    
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
    """
    
    def __init__(
        self,
        root_dir: Union[str, Path],
        transform: Optional[callable] = None,
        chord_filter: Optional[List[str]] = None,
        position_filter: Optional[List[str]] = None,
        technique_filter: Optional[List[str]] = None,
    ):
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
        Priority: _upload_to_hf -> root_dir -> download
        """
        # Priority 1: Check _upload_to_hf (ML downloaded data)
        upload_dir = self.root_dir.parent.parent / "_upload_to_hf" / "wav_files"
        scan_dir = None
        
        if upload_dir.exists():
            wav_count = len(list(upload_dir.rglob("*.wav")))
            if wav_count >= 660:
                scan_dir = upload_dir
                logger.info(f"Using _upload_to_hf data: {upload_dir} ({wav_count} files)")
        
        # Priority 2: Use root_dir (dataset/raw)
        if scan_dir is None:
            if not self.root_dir.exists():
                raise FileNotFoundError(f"Dataset directory not found: {self.root_dir}")
            
            wav_files = list(self.root_dir.rglob("*.wav"))
            
            # Priority 3: Download if empty
            if not wav_files:
                _download_from_hf(self.root_dir)
                wav_files = list(self.root_dir.rglob("*.wav"))
            
            scan_dir = self.root_dir
            logger.info(f"Using root_dir data: {scan_dir} ({len(wav_files)} files)")
        
        # Load samples from scan_dir
        wav_files = list(scan_dir.rglob("*.wav"))
        logger.info(f"Found {len(wav_files)} WAV files in {scan_dir}")
        
        skipped_count = 0
        
        for wav_path in wav_files:
            jams_path = wav_path.with_suffix('.jams')
            
            if not jams_path.exists():
                warnings.warn(
                    f"Missing JAMS annotation for {wav_path.name}, skipping. "
                    f"Expected: {jams_path.name}"
                )
                skipped_count += 1
                continue
            
            try:
                metadata = load_jams_annotation(jams_path)
                
                if self.chord_filter and metadata['chord_name'] not in self.chord_filter:
                    continue
                if self.position_filter and metadata['position'] not in self.position_filter:
                    continue
                if self.technique_filter and metadata['technique'] not in self.technique_filter:
                    continue
                
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
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> ChordSample:
        sample = self.samples[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample
    
    def get_by_chord(self, chord_name: str) -> List[ChordSample]:
        return [s for s in self.samples if s.chord_name == chord_name]
    
    def get_chord_distribution(self) -> Dict[str, int]:
        distribution = {}
        for sample in self.samples:
            distribution[sample.chord_name] = distribution.get(sample.chord_name, 0) + 1
        return dict(sorted(distribution.items()))
    
    def statistics(self) -> Dict[str, Any]:
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
