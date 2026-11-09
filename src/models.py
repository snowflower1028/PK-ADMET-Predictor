"""2주차 모델: MLP와 GIN.

MLP는 1주차와 같은 입력(descriptor+fingerprint)을 받는다. 입력이 같으니
RandomForest와 공정하게 비교된다. 숫자가 떨어진다면 그건 "신경망이
이 크기 데이터에서 트리 모델을 못 이긴다"는 결과이고, 그것도 결과다.

GIN은 그래프를 직접 받는다. Xu et al. 2019 "How Powerful are Graph
Neural Networks?"(ICLR)의 구조를 이 데이터에 맞춰 구현했다. 원 논문과
다른 점은 결합(엣지) 특징을 메시지에 더한 것으로, 분자에서는 단일결합과
이중결합이 다르게 취급돼야 하기 때문이다.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .graph import ATOM_DIM, BOND_DIM


class MLP(nn.Module):
    """descriptor+fingerprint 벡터 -> 스칼라 예측.

    fingerprint가 2048비트라 입력이 2075차원으로 크다. 데이터는 수백~수천
    개뿐이므로 과적합을 막는 쪽으로 간다: 층을 깊게 하지 않고, BatchNorm과
    Dropout을 넣는다.
    """

    def __init__(self, n_features: int, hidden: int = 512, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class GINLayer(nn.Module):
    """GIN 한 층. 이웃 원자의 벡터를 '합'으로 모은다.

    합을 쓰는 이유가 GIN의 요점이다. 평균이나 최댓값으로 모으면 이웃이
    3개인 원자와 6개인 원자를 구분하지 못하는 경우가 생긴다. 합은 개수
    정보를 보존해서 더 많은 그래프를 구별할 수 있다.

    h_v <- MLP( (1 + eps) * h_v + sum_{u in N(v)} ReLU(h_u + W e_uv) )

    eps는 학습 가능한 스칼라로, 자기 자신을 이웃보다 얼마나 더 볼지를
    데이터가 정하게 한다.
    """

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.eps = nn.Parameter(torch.zeros(1))
        self.edge_proj = nn.Linear(BOND_DIM, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 2 * dim),
            nn.BatchNorm1d(2 * dim),
            nn.ReLU(),
            nn.Linear(2 * dim, dim),
        )
        self.norm = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, edge_index, edge_attr):
        # h: (원자 수, dim)
        if edge_index.shape[1] > 0:
            src, dst = edge_index[0], edge_index[1]
            # 보내는 쪽 원자 벡터에 결합 정보를 더해 메시지를 만든다
            messages = torch.relu(h[src] + self.edge_proj(edge_attr))
            # dst가 같은 메시지끼리 더한다. index_add_가 곧 scatter 합이다.
            aggregated = torch.zeros_like(h)
            aggregated.index_add_(0, dst, messages)
        else:
            aggregated = torch.zeros_like(h)

        out = self.mlp((1.0 + self.eps) * h + aggregated)
        # 잔차 연결: 층을 쌓아도 초반 정보가 씻겨 나가지 않게 한다
        return self.dropout(torch.relu(self.norm(out)) + h)


class GIN(nn.Module):
    """원자 그래프 -> 스칼라 예측.

    층 수가 곧 '몇 결합 떨어진 원자까지 보는가'이다. 3층이면 3결합
    이웃까지 본다. 분자가 작아서(대부분 원자 20~40개) 층을 더 쌓으면
    모든 원자가 비슷해지는 over-smoothing이 생긴다.
    """

    def __init__(
        self,
        dim: int = 128,
        n_layers: int = 3,
        dropout: float = 0.1,
        readout: str = "sum",
        n_desc: int = 0,
    ):
        super().__init__()
        self.embed = nn.Linear(ATOM_DIM, dim)
        self.layers = nn.ModuleList(
            [GINLayer(dim, dropout) for _ in range(n_layers)]
        )
        self.readout = readout
        self.n_desc = n_desc

        # n_desc > 0이면 분자 단위 descriptor를 그래프 임베딩에 이어 붙인다.
        # 2주차 결과의 동기: 그래프만 보는 GIN은 구조가 지배하는 Clearance에서
        # 가장 셌지만, 분자 전체의 물리화학적 성질이 지배하는 Caco2에서
        # RandomForest보다 확실히 나빴다. 그 성질을 직접 넣어 준다.
        # descriptor는 분자 하나당 한 벡터라서 원자 단위가 아니라 readout
        # 뒤에 붙이는 것이 자연스럽다.
        if n_desc > 0:
            self.desc_norm = nn.BatchNorm1d(n_desc)
        head_in = dim + n_desc
        self.head = nn.Sequential(
            nn.Linear(head_in, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 1),
        )

    def forward(self, atom_x, edge_index, edge_attr, batch_idx, desc=None):
        h = self.embed(atom_x)
        for layer in self.layers:
            h = layer(h, edge_index, edge_attr)

        # 원자별 벡터를 분자별로 모은다(readout). batch_idx가 어느 원자가
        # 어느 분자인지 알려 준다.
        n_mols = int(batch_idx.max().item()) + 1 if batch_idx.numel() else 0
        pooled = torch.zeros(n_mols, h.shape[1], device=h.device, dtype=h.dtype)
        pooled.index_add_(0, batch_idx, h)
        if self.readout == "mean":
            counts = torch.zeros(n_mols, device=h.device, dtype=h.dtype)
            counts.index_add_(0, batch_idx, torch.ones_like(batch_idx, dtype=h.dtype))
            pooled = pooled / counts.clamp(min=1).unsqueeze(-1)

        if self.n_desc > 0:
            if desc is None or desc.shape[1] != self.n_desc:
                raise ValueError(
                    f"descriptor {self.n_desc}열을 기대했는데 "
                    f"{None if desc is None else desc.shape[1]}열이 들어왔다"
                )
            pooled = torch.cat([pooled, self.desc_norm(desc)], dim=1)

        return self.head(pooled).squeeze(-1)


def make_model(kind: str, n_features: int | None = None, **kwargs) -> nn.Module:
    if kind == "mlp":
        if n_features is None:
            raise ValueError("MLP에는 n_features가 필요하다")
        return MLP(n_features, **kwargs)
    if kind == "gin":
        # 그래프만 본다
        return GIN(n_desc=0, **kwargs)
    if kind == "gin+desc":
        if not n_features:
            raise ValueError("gin+desc에는 descriptor 개수(n_features)가 필요하다")
        return GIN(n_desc=n_features, **kwargs)
    raise ValueError(f"알 수 없는 모델: {kind}")
