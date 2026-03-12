"""
OpenGuitarChordProject - 推理脚本
使用方法:
    # 单文件
    python ml/predict.py --wav dataset/raw/C/open-down-enya-direct-001.wav

    # 批量测试整个文件夹
    python ml/predict.py --wav_dir dataset/raw/Am
"""

import argparse
from pathlib import Path

import torch
import torchaudio.transforms as T
import soundfile as sf

from dataset import SAMPLE_RATE, N_MELS, N_FFT, HOP_LENGTH, TARGET_LENGTH, IDX_TO_CHORD
from model import get_model

# ── 特征提取（与 dataset.py 保持一致）────────────────────────────

mel_transform = T.MelSpectrogram(
    sample_rate=SAMPLE_RATE, n_fft=N_FFT,
    hop_length=HOP_LENGTH, n_mels=N_MELS
)
db_transform = T.AmplitudeToDB(top_db=80)


def wav_to_mel(wav_path: str) -> torch.Tensor:
    data, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
    waveform = torch.from_numpy(data.T)  # [channels, samples]

    if sr != SAMPLE_RATE:
        import torchaudio
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    mel = mel_transform(waveform)
    mel = db_transform(mel)

    t = mel.shape[-1]
    if t < TARGET_LENGTH:
        mel = torch.nn.functional.pad(mel, (0, TARGET_LENGTH - t))
    else:
        mel = mel[..., :TARGET_LENGTH]

    mel = (mel - mel.mean()) / (mel.std() + 1e-8)
    return mel.unsqueeze(0)  # [1, 1, n_mels, T]


# ── 推理函数 ─────────────────────────────────────────────────────

def predict_one(wav_path: str, model, device) -> dict:
    mel = wav_to_mel(wav_path).to(device)
    with torch.no_grad():
        logits = model(mel)
        probs  = torch.softmax(logits, dim=1)[0]
    top3 = probs.topk(3)
    return {
        "file":  Path(wav_path).name,
        "pred":  IDX_TO_CHORD[top3.indices[0].item()],
        "conf":  top3.values[0].item(),
        "top3":  [(IDX_TO_CHORD[i.item()], v.item())
                  for i, v in zip(top3.indices, top3.values)],
    }


def load_model(ckpt_path: str, device) -> torch.nn.Module:
    ckpt  = torch.load(ckpt_path, map_location=device)
    model = get_model("small", num_classes=14).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"已加载模型 (epoch {ckpt['epoch']}, val_acc={ckpt['val_acc']*100:.2f}%)")
    return model


# ── CLI ──────────────────────────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_model(args.ckpt, device)

    if args.wav:
        result = predict_one(args.wav, model, device)
        print(f"\n文件: {result['file']}")
        print(f"预测: {result['pred']}  (置信度: {result['conf']*100:.1f}%)")
        print("Top-3:")
        for chord, prob in result["top3"]:
            bar = "█" * int(prob * 30)
            print(f"  {chord:>4}: {prob*100:5.1f}%  {bar}")

    elif args.wav_dir:
        wav_files = sorted(Path(args.wav_dir).glob("*.wav"))
        print(f"\n批量预测 {len(wav_files)} 个文件 (目录: {args.wav_dir})")
        print(f"{'文件名':<45} {'预测':>5}  {'置信度':>7}")
        print("-" * 62)
        correct, total = 0, 0
        for wav in wav_files:
            r = predict_one(str(wav), model, device)
            true_chord = wav.parent.name
            mark = "✓" if r["pred"] == true_chord else "✗"
            print(f"{wav.name:<45} {r['pred']:>5}  {r['conf']*100:>6.1f}%  {mark}")
            if true_chord in IDX_TO_CHORD.values():
                total += 1
                if r["pred"] == true_chord:
                    correct += 1
        if total > 0:
            print(f"\n准确率: {correct}/{total} = {correct/total*100:.1f}%")

    else:
        print("请指定 --wav 或 --wav_dir")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OGCP 和弦识别推理")
    parser.add_argument("--wav",     type=str, help="单个 WAV 文件路径")
    parser.add_argument("--wav_dir", type=str, help="WAV 文件夹（批量）")
    parser.add_argument("--ckpt",    type=str, default="models/best_model.pth")
    args = parser.parse_args()
    main(args)
