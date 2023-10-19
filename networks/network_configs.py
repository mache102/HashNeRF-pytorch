from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class SigmaNetConfig:
    input_ch: int = 3
    layers: int = 3
    hdim: int = 64
    geo_feat_dim: int = 15
    skips: List[int] = field(default_factory=lambda: [4])

@dataclass
class ColorNetConfig:
    input_ch: int = 3
    layers: int = 4
    hdim: int = 64
