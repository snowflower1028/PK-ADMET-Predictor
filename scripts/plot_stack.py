"""스태킹 결과 그림. results/stack_ablation.png을 만든다.

  python scripts/plot_stack.py

읽는 파일
  results/baseline.csv               1주차 트리 모델 (train+valid 학습)
  results/models.csv                 2주차 신경망
  results/stack.csv                  4주차 스태킹
  results/stack_meta_diagnostics.csv 1단 train 특징의 정확도

그림은 두 줄이다.

  윗줄 (과제 3칸)  사다리. 1주차부터 4주차까지 같은 축에 올려 본다.
  아랫줄 (과제 3칸) 2x2 교호작용. x축은 1단 특징을 OOF로 만들었는지 누출된
                   방식으로 만들었는지, 선 두 개는 residual head를 켰는지다.
                   x축 라벨에 "그 특징이 train에서 얼마나 정답에 가까운지"를
                   같이 적어 둔다 — 누출 쪽이 더 정확해 **보이는데** test
                   점수는 그 반대라는 것이 이 그림의 요점이다.

라벨은 영문으로 둔다. matplotlib 기본 폰트에 한글이 없어서 저장된 PNG에
네모칸이 찍히는 것을 막는다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import RESULTS_DIR, TASKS  # noqa: E402

# (표시 이름, 출처 CSV, 그 CSV에서 고를 행, 색 분류)
#
# 색 분류
#   ref   1~2주차 기준선 (회색)
#   base  이번 실험의 1단 기반모델, train만 학습 (연한 파랑)
#   new   이번 실험의 2단 모델, OOF 특징 (파랑)
#   leak  누출 특징 대조군 (빨강)
LADDER = [
    ("RF desc+fp (wk1, train+valid)", "baseline", ("rf", "desc+fp"), "ref"),
    ("LightGBM desc+fp (wk1)", "baseline", ("lgbm", "desc+fp"), "ref"),
    ("RF (stage-1, train only)", "stack", "rf-base", "base"),
    ("LightGBM (stage-1, train only)", "stack", "lgbm-base", "base"),
    ("MLP desc+fp (wk2)", "models", "mlp", "ref"),
    ("GIN (wk2, graph only)", "models", "gin", "ref"),
    ("GIN+desc (wk2)", "models", "gin+desc", "ref"),
    ("GIN+meta (graph+RF/LGBM)", "stack", "gin+meta", "new"),
    ("GIN+stack, residual head", "stack", "gin+stack", "new"),
    ("GIN+stack, plain head", "stack", "gin+stack-plain", "new"),
    ("GIN+stack, residual + LEAKY", "stack", "gin+stack-infold", "leak"),
    ("GIN+stack, plain + LEAKY", "stack", "gin+stack-plain-infold", "leak"),
]

COLORS = {"ref": "0.72", "base": "#9ecae1", "new": "#1f6fb4", "leak": "#c0392b"}

# 아랫줄 2x2 교호작용에서 쓸 네 칸
CELLS = {
    ("oof", True): "gin+stack",
    ("infold", True): "gin+stack-infold",
    ("oof", False): "gin+stack-plain",
    ("infold", False): "gin+stack-plain-infold",
}


def load(name: str) -> pd.DataFrame:
    path = RESULTS_DIR / f"{name}.csv"
    if not path.exists():
        raise SystemExit(f"없는 파일: {path}. 먼저 해당 스크립트를 돌려야 한다.")
    return pd.read_csv(path)


def pick(frames: dict, task: str, source: str, key) -> tuple:
    """사다리 한 칸의 (mean, std). 없으면 (nan, nan)."""
    df = frames[source]
    d = df[df["task"] == task]
    if source == "baseline":
        model, feat = key
        d = d[(d["model"] == model) & (d["features"] == feat)]
    else:
        d = d[d["model"] == key]
    if not len(d):
        return float("nan"), float("nan")
    r = d.iloc[0]
    return float(r["mean"]), float(r["std"])


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frames = {n: load(n) for n in ("baseline", "models", "stack")}
    diag = load("stack_meta_diagnostics")

    tasks = [t for t in TASKS if (frames["stack"]["task"] == t).any()]
    fig, axes = plt.subplots(2, len(tasks), figsize=(6.0 * len(tasks), 11.0))
    if len(tasks) == 1:
        axes = axes.reshape(2, 1)

    # ---------- 윗줄: 사다리 ------------------------------------------------
    for ax, task in zip(axes[0], tasks):
        metric = TASKS[task]["metric"]
        higher = TASKS[task]["higher_is_better"]

        labels, means, stds, colors = [], [], [], []
        for label, source, key, cls in LADDER:
            m, s = pick(frames, task, source, key)
            if not np.isfinite(m):
                continue
            labels.append(label)
            means.append(m)
            stds.append(s)
            colors.append(COLORS[cls])

        y = np.arange(len(labels))[::-1]  # 위에서 아래로 사다리 순서
        ax.barh(y, means, xerr=stds, height=0.68, color=colors,
                error_kw=dict(ecolor="0.25", lw=1.1, capsize=3))

        # 2주차 최고 기록을 기준선으로 그어 둔다. 4주차가 이 선을 넘었는가가
        # 이 그림에서 읽어야 할 첫 번째 질문이다.
        ref = dict(zip(labels, means)).get("GIN+desc (wk2)")
        if ref is not None and np.isfinite(ref):
            ax.axvline(ref, color="0.35", lw=1.2, ls="--",
                       label="GIN+desc (wk2 best NN)")
            ax.legend(fontsize=7.5, loc="lower right")

        span = max(np.array(means) + np.nan_to_num(np.array(stds)))
        for yi, (m, s) in zip(y, zip(means, stds)):
            ax.text(m + span * 0.015, yi, f" {m:.3f}+/-{s:.3f}",
                    va="center", fontsize=7.5)

        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel(f"{metric}  ({'higher' if higher else 'lower'} is better)")
        ax.set_title(task, fontsize=11)
        ax.set_xlim(0, span * 1.30)
        ax.grid(alpha=0.25, axis="x")

    # ---------- 아랫줄: 2x2 교호작용 ----------------------------------------
    piv = diag.pivot_table(index="task", columns="meta_mode",
                           values="train_feature_mae_in_sd", aggfunc="mean")
    st = frames["stack"]

    for ax, task in zip(axes[1], tasks):
        metric = TASKS[task]["metric"]
        higher = TASKS[task]["higher_is_better"]
        modes = ["oof", "infold"]
        x = np.arange(len(modes))

        for residual, style in ((True, dict(marker="o", ls="-", color="#c0392b")),
                                (False, dict(marker="s", ls="-", color="#1f6fb4"))):
            vals, errs = [], []
            for mode in modes:
                m, s = pick(frames, task, "stack", CELLS[(mode, residual)])
                vals.append(m)
                errs.append(s)
            ax.errorbar(x, vals, yerr=errs, lw=2, capsize=4, markersize=7,
                        label=("residual head ON" if residual
                               else "residual head OFF (plain concat)"),
                        **style)
            for xi, v in zip(x, vals):
                if np.isfinite(v):
                    ax.annotate(f"{v:.3f}", (xi, v), textcoords="offset points",
                                xytext=(0, 9), ha="center", fontsize=8)

        ref, _ = pick(frames, task, "models", "gin+desc")
        if np.isfinite(ref):
            ax.axhline(ref, color="0.35", lw=1.2, ls="--",
                       label=f"GIN+desc (wk2) = {ref:.3f}")

        # x축 라벨에 "그 특징이 train에서 얼마나 정답에 가까운지"를 적는다
        ticks = []
        for mode in modes:
            try:
                acc = float(piv.loc[task, mode])
                note = f"\nstage-1 train error\n{acc:.2f} SD"
            except (KeyError, ValueError):
                note = ""
            ticks.append(("out-of-fold" if mode == "oof"
                          else "in-fold (LEAKY)") + note)
        ax.set_xticks(x)
        ax.set_xticklabels(ticks, fontsize=8.5)
        ax.set_xlim(-0.35, len(modes) - 0.65)
        ax.set_ylabel(f"test {metric}  ({'higher' if higher else 'lower'} is better)")
        ax.set_title(f"{task}\nhow the meta-features were built", fontsize=10)
        ax.grid(alpha=0.25, axis="y")
        ax.legend(fontsize=7.5, loc="best")

    fig.suptitle(
        "Two-stage stacking: RF / LightGBM predictions as GIN input features"
        "   —   TDC ADMET, scaffold split, 5 seeds, TDC evaluate_many()",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    out = RESULTS_DIR / "stack_ablation.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
