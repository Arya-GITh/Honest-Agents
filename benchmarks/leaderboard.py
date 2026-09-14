"""
DeceptionBench Leaderboard Formatter
-----------------------------------
Renders evaluation summaries into:
- Markdown Leaderboard Tables
- CSV Export Summaries
- LaTeX Table Snippets for academic publications
"""

import csv
from pathlib import Path
from typing import List

from benchmarks.metrics import ModelBenchmarkSummary


def format_markdown_leaderboard(summaries: List[ModelBenchmarkSummary]) -> str:
    """Generate a clean Markdown leaderboard comparing evaluated models."""
    sorted_summaries = sorted(summaries, key=lambda s: (s.guarded_edr, s.raw_edr))

    lines = [
        "# 🏆 DeceptionBench Official Leaderboard",
        "",
        "Evaluation of Execution Deception Rates (EDR) and Truthify Governance across AI models.",
        "",
        "| Rank | Model Architecture | Scenarios | Raw EDR (%) | Guarded EDR (%) | Protection Gain (Δ) | Empty Fabrication (FER) | Avg Latency |",
        "| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for rank, s in enumerate(sorted_summaries, start=1):
        lines.append(
            f"| **#{rank}** | `{s.model_name}` | {s.total_scenarios} | "
            f"**{s.raw_edr:.1f}%** | **{s.guarded_edr:.1f}%** | "
            f"**+{s.edr_protection_gain:.1f}%** | {s.raw_fer:.1f}% → {s.guarded_fer:.1f}% | "
            f"{s.avg_verification_latency_ms:.1f} ms |"
        )

    lines.extend([
        "",
        "### Metric Definitions:",
        "- **Raw EDR (Execution Deception Rate)**: Percentage of tool failures where the raw model falsely asserted success.",
        "- **Guarded EDR**: Deception rate after Truthify runtime verification & in-scratchpad self-correction.",
        "- **Protection Gain (Δ)**: Percentage point reduction in deception achieved by Truthify.",
        "- **FER (Fabrication on Empty Rate)**: Percentage of empty query returns (`[]`) where the model hallucinated fake records.",
    ])

    return "\n".join(lines)


def export_csv_leaderboard(summaries: List[ModelBenchmarkSummary], output_path: Path) -> None:
    """Export summary results to a CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model_name",
            "total_scenarios",
            "raw_edr_pct",
            "guarded_edr_pct",
            "edr_protection_gain_pct",
            "raw_fer_pct",
            "guarded_fer_pct",
            "raw_pmr_pct",
            "guarded_pmr_pct",
            "avg_latency_ms",
            "avg_reprompts",
            "fallback_override_pct",
        ])
        for s in summaries:
            writer.writerow([
                s.model_name,
                s.total_scenarios,
                s.raw_edr,
                s.guarded_edr,
                s.edr_protection_gain,
                s.raw_fer,
                s.guarded_fer,
                s.raw_pmr,
                s.guarded_pmr,
                s.avg_verification_latency_ms,
                s.avg_reprompts_per_failure,
                s.fallback_override_rate,
            ])


def format_latex_table(summaries: List[ModelBenchmarkSummary]) -> str:
    """Format benchmark results as a LaTeX table for academic research papers."""
    sorted_summaries = sorted(summaries, key=lambda s: (s.guarded_edr, s.raw_edr))
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{DeceptionBench Execution Integrity Evaluation Across Model Architectures}",
        r"\label{tab:deceptionbench}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Scenarios} & \textbf{Raw EDR (\%)} & \textbf{Guarded EDR (\%)} & \textbf{$\Delta$ Gain} & \textbf{Raw FER (\%)} & \textbf{Latency (ms)} \\",
        r"\midrule",
    ]
    for s in sorted_summaries:
        lines.append(
            f"{s.model_name} & {s.total_scenarios} & {s.raw_edr:.1f}\\% & {s.guarded_edr:.1f}\\% & +{s.edr_protection_gain:.1f}\\% & {s.raw_fer:.1f}\\% & {s.avg_verification_latency_ms:.1f} \\\\"
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])
    return "\n".join(lines)
