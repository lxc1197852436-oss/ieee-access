from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "outputs" / "midterm" / "midterm_model_comparison.csv"
SVG_PATH = ROOT / "outputs" / "midterm" / "midterm_model_comparison.svg"


def read_rows() -> list[dict]:
    with CSV_PATH.open(encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def scale(values: list[float], width: float) -> list[float]:
    min_v = min(values)
    max_v = max(values)
    span = max(max_v - min_v, 1e-9)
    return [40 + (v - min_v) / span * width for v in values]


def main() -> None:
    rows = read_rows()
    names = [r["policy"] for r in rows]
    rewards = [float(r["total_reward_yuan"]) for r in rows]
    cvars = [float(r["cvar_5_yuan"]) for r in rows]
    reward_widths = scale(rewards, 520)
    cvar_widths = scale(cvars, 520)

    y0 = 82
    row_h = 54
    height = y0 + len(rows) * row_h + 80
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="980" height="{height}" viewBox="0 0 980 {height}">',
        '<rect width="980" height="100%" fill="#f7f9fc"/>',
        '<text x="32" y="36" font-size="22" font-weight="700" fill="#172033">中期实验模型对比</text>',
        '<text x="32" y="60" font-size="13" fill="#5c667a">广东公开数据校准的VPP样例场景；数值越接近0表示亏损更小，CVaR越接近0表示尾部风险更低。</text>',
        '<text x="190" y="80" font-size="12" fill="#5c667a">总收益</text>',
        '<text x="650" y="80" font-size="12" fill="#5c667a">CVaR 5%</text>',
    ]
    for i, name in enumerate(names):
        y = y0 + i * row_h
        parts.append(f'<text x="32" y="{y + 28}" font-size="14" font-weight="600" fill="#172033">{name}</text>')
        parts.append(f'<rect x="190" y="{y + 9}" width="{reward_widths[i]:.1f}" height="18" rx="4" fill="#2155a3"/>')
        parts.append(f'<text x="{200 + reward_widths[i]:.1f}" y="{y + 24}" font-size="12" fill="#172033">{rewards[i]:.0f}</text>')
        parts.append(f'<rect x="650" y="{y + 9}" width="{cvar_widths[i]:.1f}" height="18" rx="4" fill="#0f8b8d"/>')
        parts.append(f'<text x="{660 + cvar_widths[i]:.1f}" y="{y + 24}" font-size="12" fill="#172033">{cvars[i]:.0f}</text>')
    parts.append(f'<text x="32" y="{height - 28}" font-size="12" fill="#5c667a">注：Soft-Q为中期阶段最大熵RL过渡基线，正式论文将升级为连续SAC/LE-DRL。</text>')
    parts.append("</svg>")
    SVG_PATH.write_text("\n".join(parts), encoding="utf-8")
    print(f"Saved: {SVG_PATH}")


if __name__ == "__main__":
    main()

