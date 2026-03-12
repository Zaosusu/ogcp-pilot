"""
OpenGuitarChordProject - CNN 模型
输入: Mel 频谱图 [B, 1, 128, 128]
输出: 14 类和弦 logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Conv → BN → ReLU → (可选) MaxPool"""
    def __init__(self, in_ch, out_ch, pool=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2, 2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ChordCNN(nn.Module):
    """
    轻量 CNN，专为 660 样本小数据集设计
    参数量 ~500K，不易过拟合
    """
    def __init__(self, num_classes: int = 14, dropout: float = 0.5):
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(1,  32, pool=True),   # [B,  32, 64, 64]
            ConvBlock(32, 64, pool=True),   # [B,  64, 32, 32]
            ConvBlock(64, 128, pool=True),  # [B, 128, 16, 16]
            ConvBlock(128, 128, pool=True), # [B, 128,  8,  8]
        )

        self.gap = nn.AdaptiveAvgPool2d(1)  # [B, 128, 1, 1]

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x


class ChordCNNLarge(nn.Module):
    """
    更大的模型，适合数据增强后使用
    基于 EfficientNet 思路的深层 CNN
    """
    def __init__(self, num_classes: int = 14, dropout: float = 0.4):
        super().__init__()

        def make_block(in_ch, out_ch, stride=1):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.GELU(),
            )

        self.stem = make_block(1, 32, stride=2)

        self.stage1 = nn.Sequential(make_block(32, 64),  make_block(64, 64))
        self.down1  = make_block(64, 64, stride=2)

        self.stage2 = nn.Sequential(make_block(64, 128), make_block(128, 128))
        self.down2  = make_block(128, 128, stride=2)

        self.stage3 = nn.Sequential(make_block(128, 256), make_block(256, 256))
        self.down3  = make_block(256, 256, stride=2)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.down1(self.stage1(x))
        x = self.down2(self.stage2(x))
        x = self.down3(self.stage3(x))
        return self.head(x)


def get_model(name: str = "small", num_classes: int = 14, **kwargs) -> nn.Module:
    """工厂函数"""
    if name == "small":
        return ChordCNN(num_classes=num_classes, **kwargs)
    elif name == "large":
        return ChordCNNLarge(num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"未知模型: {name}，可选 'small' | 'large'")


if __name__ == "__main__":
    # 快速验证
    model = get_model("small")
    x = torch.randn(4, 1, 128, 128)
    out = model(x)
    print(f"输出形状: {out.shape}")  # [4, 14]

    total = sum(p.numel() for p in model.parameters())
    print(f"参数量: {total:,}")
