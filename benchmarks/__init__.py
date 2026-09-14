"""
DeceptionBench Evaluation Suite
-------------------------------
Standardized evaluation benchmark, metrics engine, and leaderboard generator
for measuring and mitigating the Execution-Claim Gap across AI agents.
"""

from benchmarks.datasets.generator import BenchmarkScenario, generate_all_scenarios, load_dataset
from benchmarks.evaluator import DeceptionBenchEvaluator
from benchmarks.leaderboard import export_csv_leaderboard, format_latex_table, format_markdown_leaderboard
from benchmarks.metrics import (
    BenchmarkCategory,
    CategoryMetrics,
    ModelBenchmarkSummary,
    ScenarioExecution,
)
from benchmarks.models.base import BaseModelProvider
from benchmarks.models.gemini_provider import GeminiModelProvider
from benchmarks.models.mock_provider import MockModelProvider
from benchmarks.models.ollama_provider import OllamaModelProvider

__all__ = [
    "BenchmarkScenario",
    "generate_all_scenarios",
    "load_dataset",
    "DeceptionBenchEvaluator",
    "format_markdown_leaderboard",
    "export_csv_leaderboard",
    "format_latex_table",
    "BenchmarkCategory",
    "CategoryMetrics",
    "ModelBenchmarkSummary",
    "ScenarioExecution",
    "BaseModelProvider",
    "MockModelProvider",
    "OllamaModelProvider",
    "GeminiModelProvider",
]
