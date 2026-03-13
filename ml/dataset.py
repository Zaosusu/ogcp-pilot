"""
OpenGuitarChordProject - Dataset Loader
使用 soundfile 读取音频，无需 FFmpeg / torchcodec
数据增强包含模拟麦克风音色（噪声、频率响应、混响）
"""

import torch
import torch.nn.functional as F
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

HF_REPO_ID = "Zaosusu/ogcp-pilot"


def _download_from_hf(root_dir: Path) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError(...)
    
    import os
    import urllib.request
    # 检测 HuggingFace 是否可达，不行就用镜像
    try:
        urllib.request.urlopen("https://huggingface.co", timeout=5)
        endpoint = "https://huggingface.co"
    except Exception:
        print("HuggingFace 不可达，切换到镜像站...")
        endpoint = "https://hf-mirror.com"
        os.environ["HF_ENDPOINT"] = endpoint

    print(f"正在从 {endpoint} 下载 ({HF_REPO_ID})...")
    snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        local_dir=str(root_dir),
    )
    print(f"下载完成，文件已保存到: {root_dir}")

SAMPLE_RATE   = 44100
N_MELS        = 128
N_FFT         = 2048
HOP_LENGTH    = 512
TARGET_LENGTH = 128


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
        # 检查是否有任何 WAV 文件，没有则自动下载
        wav_files = list(self.root_dir.rglob("*.wav"))
        if not wav_files:
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
            mel = F.pad(mel, (0, TARGET_LENGTH - t))
        else:
            mel = mel[..., :TARGET_LENGTH]

        mel = (mel - mel.mean()) / (mel.std() + 1e-8)
        return mel, label

    def _augment(self, waveform):
        """
        综合增强：基础增强 + 随机模拟麦克风音色
        每次训练以 50% 概率触发麦克风模拟
        """
        # ── 基础增强（每次都做）──────────────────────────────
        # 随机音量 ±4dB
        gain = 10 ** (torch.FloatTensor(1).uniform_(-4, 4) / 20)
        waveform = waveform * gain

        # 随机时移（最多 0.1s）
        shift = int(torch.randint(0, int(0.1 * SAMPLE_RATE), (1,)).item())
        waveform = torch.roll(waveform, shift, dims=-1)

        # ── 麦克风模拟（50% 概率触发）───────────────────────
        if torch.rand(1).item() < 0.5:
            waveform = self._simulate_mic(waveform)

        return waveform

    def _simulate_mic(self, waveform):
        """
        模拟手机/麦克风录音的三种效果：
        1. 高斯白噪声（模拟底噪）
        2. 低频滚降（手机麦频率响应）
        3. 简单混响（房间反射）
        """
        # 1. 加噪声（SNR 随机 20~40dB）
        snr_db  = torch.FloatTensor(1).uniform_(30, 40).item()
        sig_pwr = waveform.pow(2).mean().clamp(min=1e-8)
        noise_pwr = sig_pwr / (10 ** (snr_db / 10))
        noise = torch.randn_like(waveform) * noise_pwr.sqrt()
        waveform = waveform + noise

        # 2. 低频滚降：简单高通模拟（手机麦在 100Hz 以下衰减）
        # 用一阶差分近似高通
        if torch.rand(1).item() < 0.5:
            alpha = torch.FloatTensor(1).uniform_(0.02, 0.08).item()
            filtered = torch.zeros_like(waveform)
            filtered[..., 0] = waveform[..., 0]
            for i in range(1, waveform.shape[-1]):
                filtered[..., i] = (1 - alpha) * filtered[..., i-1] + \
                                   (1 - alpha) * (waveform[..., i] - waveform[..., i-1])
            waveform = filtered

        # 3. 简单混响（延迟叠加模拟房间反射）
        if torch.rand(1).item() < 0.5:
            delay_ms  = torch.FloatTensor(1).uniform_(20, 80).item()
            delay_smp = int(delay_ms * SAMPLE_RATE / 1000)
            decay     = torch.FloatTensor(1).uniform_(0.1, 0.2).item()
            reverb    = torch.zeros_like(waveform)
            reverb[..., delay_smp:] = waveform[..., :-delay_smp] * decay
            waveform = waveform + reverb

        # 重新归一化防止裁切
        peak = waveform.abs().max().clamp(min=1e-8)
        if peak > 1.0:
            waveform = waveform / peak

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
