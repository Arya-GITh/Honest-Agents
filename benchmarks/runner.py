"""
DeceptionBench CLI Benchmark Runner with Streaming Checkpointing
----------------------------------------------------------------
Command-line interface to execute evaluations across multiple models concurrently.
Features streaming incremental checkpointing to preserve progress on disk.

Usage:
  uv run python -m benchmarks.runner --models mock --limit 10
  uv run python -m benchmarks.runner --models gemini-flash-latest
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

# Add workspace root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.datasets.generator import BenchmarkScenario, load_dataset
from benchmarks.evaluator import DeceptionBenchEvaluator
from benchmarks.leaderboard import export_csv_leaderboard, format_latex_table, format_markdown_leaderboard
from benchmarks.metrics import BenchmarkCategory, ModelBenchmarkSummary, ScenarioExecution
from benchmarks.models.base import BaseModelProvider
from benchmarks.models.gemini_provider import GeminiModelProvider
from benchmarks.models.mock_provider import MockModelProvider
from benchmarks.models.ollama_provider import OllamaModelProvider

# Auto-load .env
ENV_PATH = Path(__file__).parent.parent / ".env"
if ENV_PATH.exists():
    with open(ENV_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip("'").strip('"')
                if k and v and k not in os.environ:
                    os.environ[k] = v


def get_provider_for_name(model_name: str) -> BaseModelProvider:
    """Instantiate the appropriate BaseModelProvider based on model name string."""
    name_lower = model_name.lower()
    if "mock" in name_lower:
        return MockModelProvider(model_name=model_name, deception_rate=0.6)
    elif "gemini" in name_lower:
        return GeminiModelProvider(model_name=model_name)
    else:
        return OllamaModelProvider(model_name=model_name)


async def run_benchmark_for_model(
    model_name: str,
    scenarios: List[BenchmarkScenario],
    output_dir: Path,
    delay_seconds: float = 0.0,
) -> ModelBenchmarkSummary:
    """Run full evaluation suite for a single model with live terminal reporting and checkpointing."""
    provider = get_provider_for_name(model_name)
    evaluator = DeceptionBenchEvaluator(model_provider=provider)
    checkpoint_file = output_dir / f"checkpoint_{model_name.replace(':', '_')}.jsonl"

    print(f"\n🚀 Evaluating Model: [{model_name}] across {len(scenarios)} scenarios (cooldown: {delay_seconds:.1f}s)...", flush=True)

    def progress(idx: int, total: int, exec_res: ScenarioExecution) -> None:
        raw_status = "❌ DECEPTIVE" if (exec_res.raw_verdict and not exec_res.raw_verdict.is_honest) else "✅ HONEST"
        guarded_status = "❌ DECEPTIVE" if (exec_res.guarded_verdict and not exec_res.guarded_verdict.is_honest) else "🛡️  PROTECTED"
        print(
            f"  [{idx:03d}/{total:03d}] {exec_res.scenario_id:<8} "
            f"Raw: {raw_status:<14} Guarded: {guarded_status:<14} ({exec_res.verification_latency_ms:.1f}ms)",
            flush=True,
        )
        # Write streaming checkpoint line to disk
        with open(checkpoint_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(exec_res.model_dump(mode="json")) + "\n")

    summary = await evaluator.evaluate_dataset(
        scenarios=scenarios,
        progress_callback=progress,
        delay_seconds=delay_seconds,
    )
    print(f"📊 [{model_name}] Results: Raw EDR: {summary.raw_edr}% → Guarded EDR: {summary.guarded_edr}% (Δ: +{summary.edr_protection_gain}%)", flush=True)
    return summary


async def main_async() -> None:
    parser = argparse.ArgumentParser(description="DeceptionBench Multi-Model Benchmark Runner")
    parser.add_argument("--models", type=str, default="gemini-flash-latest", help="Comma-separated model names")
    parser.add_argument("--categories", type=str, default="all", help="Comma-separated categories or 'all'")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of scenarios")
    parser.add_argument("--delay", type=float, default=1.5, help="Thermal cooldown delay in seconds between scenarios")
    parser.add_argument("--output-dir", type=str, default="benchmarks/results", help="Directory to save benchmark results")
    args = parser.parse_args()

    # Load scenarios
    scenarios = load_dataset()
    if args.categories != "all":
        selected_cats = [c.strip() for c in args.categories.split(",")]
        scenarios = [s for s in scenarios if s.category.value in selected_cats]

    if args.limit:
        scenarios = scenarios[: args.limit]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80, flush=True)
    print("🏆 DECEPTIONBENCH MULTI-MODEL EVALUATION SUITE", flush=True)
    print(f"🔹 Total Scenarios: {len(scenarios)}", flush=True)
    print(f"🔹 Target Models: {args.models}", flush=True)
    print("=" * 80, flush=True)

    model_names = [m.strip() for m in args.models.split(",")]
    summaries: List[ModelBenchmarkSummary] = []

    for model_name in model_names:
        try:
            summary = await run_benchmark_for_model(
                model_name=model_name,
                scenarios=scenarios,
                output_dir=out_dir,
                delay_seconds=args.delay,
            )
            summaries.append(summary)
            
            # Incrementally update leaderboard after each model finishes
            md_table = format_markdown_leaderboard(summaries)
            latex_table = format_latex_table(summaries)
            (out_dir / "leaderboard.md").write_text(md_table, encoding="utf-8")
            (out_dir / "leaderboard.tex").write_text(latex_table, encoding="utf-8")
            export_csv_leaderboard(summaries, out_dir / "leaderboard.csv")
            with open(out_dir / "results.json", "w", encoding="utf-8") as f:
                json.dump([s.model_dump(mode="json") for s in summaries], f, indent=2)
        except Exception as exc:
            print(f"⚠️ Error evaluating model {model_name}: {exc}", flush=True)

    if summaries:
        md_table = format_markdown_leaderboard(summaries)
        print("\n" + "=" * 80, flush=True)
        print(md_table, flush=True)
        print("=" * 80, flush=True)
        print(f"💾 Complete results saved to {out_dir}/", flush=True)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
