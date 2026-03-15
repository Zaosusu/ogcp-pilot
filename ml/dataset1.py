"""
OpenGuitarChordProject - Dataset Loader (GPU版，简化增强)
Dataset只返回原始波形，梅尔频谱在GPU上计算
"""

import torch
import torch.nn.functional as F
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np

CHORD_LABELS = [
    "A", "Am", "B", "Bm",
    "C", "Cm", "D", "Dm",
    "E", "Em", "F", "Fm",
    "G", "Gm"
]
CHORD_TO_IDX = {c: i for i, c in enumerate(CHORD_LABELS)}
IDX_TO_CHORD = {i: c for c, i in CHORD_TO_IDX.items()}

HF_REPO_ID = "Zaosusu/ogcp-pilot"

# ── 音频参数（训练和推理共用）────────────────────────
SAMPLE_RATE   = 44100
N_FFT         = 2048
HOP_LENGTH    = 512
N_MELS        = 128
TARGET_LENGTH = 128  # 时间帧数（约1.49秒）
MAX_SAMPLES   = 128 * 512  # 约1.5秒音频




def _download_from_hf(root_dir: Path) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError("huggingface_hub is required")
    import os, urllib.request
    try:
        urllib.request.urlopen("https://hf-mirror.com", timeout=5)
        endpoint = "https://hf-mirror.com"
        os.environ["HF_ENDPOINT"] = endpoint
    except Exception:
        endpoint = "https://huggingface.co"
    local_dir = root_dir.parent.parent
    print(f"正在从 {endpoint} 下载 ({HF_REPO_ID})...")
    snapshot_download(repo_id=HF_REPO_ID, repo_type="dataset", local_dir=str(local_dir))
    print(f"下载完成: {root_dir}")


class GuitarChordDataset(Dataset):
    def __init__(self, root_dir: str, split: str = "train",
                 train_ratio: float = 0.8, seed: int = 42,
                 augment: bool = False):
        self.root_dir = Path(root_dir)
        self.augment  = augment and (split == "train")

        all_samples = self._collect_samples()
        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(all_samples))
        n_train = int(len(indices) * train_ratio)
        n_val   = int(len(indices) * (1 - train_ratio) / 2)

        if split == "train":
            chosen = indices[:n_train]
        elif split == "val":
            chosen = indices[n_train:n_train + n_val]
        else:
            chosen = indices[n_train + n_val:]

        self.samples = [all_samples[i] for i in chosen]
        print(f"[{split}] {len(self.samples)} 个样本")

    def _collect_samples(self):
        self.root_dir.mkdir(parents=True, exist_ok=True)
        if len(list(self.root_dir.rglob("*.wav"))) < 660:
            _download_from_hf(self.root_dir)

        samples = []
        for chord in CHORD_LABELS:
            chord_dir = self.root_dir / chord
            if not chord_dir.exists():
                continue
            for wav_path in sorted(chord_dir.glob("*.wav")):
                samples.append((wav_path, CHORD_TO_IDX[chord]))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        wav_path, label = self.samples[idx]

        # 加载音频
        data, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
        waveform = torch.from_numpy(data.T)

        # 重采样
        if sr != SAMPLE_RATE:
            import torchaudio
            waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)

        # 转单声道
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # 基础数据增强
        if self.augment:
            waveform = self._augment(waveform)

        # 返回原始波形，不做梅尔频谱
        return waveform.squeeze(0), label

    def _augment(self, waveform):
        # 随机音量 ±3dB
        gain = 10 ** (torch.FloatTensor(1).uniform_(-3, 3) / 20)
        waveform = waveform * gain
        
        # 随机时移（最多 0.1s）
        shift = int(torch.randint(0, int(0.1 * SAMPLE_RATE), (1,)).item())
        waveform = torch.roll(waveform, shift, dims=-1)
        
        return waveform


def collate_fn(batch):
    """处理变长音频"""
    waveforms, labels = zip(*batch)
    max_len = min(max(w.shape[0] for w in waveforms), MAX_SAMPLES * 2)
    
    padded = []
    for w in waveforms:
        if w.shape[0] < max_len:
            w = F.pad(w, (0, max_len - w.shape[0]))
        else:
            w = w[:max_len]
        padded.append(w)
    
    return torch.stack(padded), torch.tensor(labels)


def get_dataloaders(root_dir: str, batch_size: int = 32, num_workers: int = 4):
    train_ds = GuitarChordDataset(root_dir, split="train", augment=True)
    val_ds   = GuitarChordDataset(root_dir, split="val",   augment=False)
    test_ds  = GuitarChordDataset(root_dir, split="test",  augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, collate_fn=collate_fn
    )

    return train_loader, val_loader, test_loader
