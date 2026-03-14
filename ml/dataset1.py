"""
OpenGuitarChordProject - Dataset Loader
使用 soundfile 读取音频，无需 FFmpeg / torchcodec
"""

import torch
import torchaudio.transforms as T
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

SAMPLE_RATE   = 44100
N_MELS        = 128
N_FFT         = 2048
HOP_LENGTH    = 512
TARGET_LENGTH = 128

HF_REPO_ID = "Zaosusu/ogcp-pilot"


def _download_from_hf(root_dir: Path) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required for auto-download. "
            "Install with: pip install huggingface_hub"
        )
    import os, urllib.request
    try:
        urllib.request.urlopen("https://hf-mirror.com", timeout=5)
        endpoint = "https://hf-mirror.com"
        os.environ["HF_ENDPOINT"] = endpoint
    except Exception:
        print("镜像站不可达，切换到官网...")
        endpoint = "https://huggingface.co"
    local_dir = root_dir.parent.parent
    print(f"正在从 {endpoint} 下载 ({HF_REPO_ID}) 到 {local_dir}...")
    snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        local_dir=str(local_dir),
    )
    print(f"下载完成，文件已保存到: {root_dir}")


class GuitarChordDataset(Dataset):
    def __init__(self, root_dir: str, split: str = "train",
                 train_ratio: float = 0.8, seed: int = 42,
                 augment: bool = False):
        self.root_dir = Path(root_dir)
        self.augment  = augment and (split == "train")
        self.mel_transform = T.MelSpectrogram(
            sample_rate=SAMPLE_RATE, n_fft=N_FFT,
            hop_length=HOP_LENGTH, n_mels=N_MELS,
        )
        self.db_transform = T.AmplitudeToDB(top_db=80)

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

        # soundfile 读取，无需 FFmpeg
        data, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
        waveform = torch.from_numpy(data.T)  # [channels, samples]

        if sr != SAMPLE_RATE:
            import torchaudio
            waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if self.augment:
            waveform = self._augment(waveform)

        mel = self.mel_transform(waveform)
        mel = self.db_transform(mel)

        t = mel.shape[-1]
        if t < TARGET_LENGTH:
            mel = torch.nn.functional.pad(mel, (0, TARGET_LENGTH - t))
        else:
            mel = mel[..., :TARGET_LENGTH]

        mel = (mel - mel.mean()) / (mel.std() + 1e-8)
        return mel, label

    def _augment(self, waveform):
        gain = 10 ** (torch.FloatTensor(1).uniform_(-3, 3) / 20)
        waveform = waveform * gain
        shift = int(torch.randint(0, int(0.1 * SAMPLE_RATE), (1,)).item())
        waveform = torch.roll(waveform, shift, dims=-1)
        return waveform


def get_dataloaders(root_dir: str, batch_size: int = 32, num_workers: int = 4):
    train_ds = GuitarChordDataset(root_dir, split="train", augment=True)
    val_ds   = GuitarChordDataset(root_dir, split="val",   augment=False)
    test_ds  = GuitarChordDataset(root_dir, split="test",  augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader
