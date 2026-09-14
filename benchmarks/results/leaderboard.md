# 🏆 DeceptionBench Official Leaderboard

Evaluation of Execution Deception Rates (EDR) and Truthify Governance across AI models.

| Rank | Model Architecture | Scenarios | Raw EDR (%) | Guarded EDR (%) | Protection Gain (Δ) | Empty Fabrication (FER) | Avg Latency |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **#1** | `qwen3:latest` | 105 | **9.5%** | **0.0%** | **+9.5%** | 4.0% → 0.0% | 1774.8 ms |
| **#2** | `gemini-flash-latest` | 105 | **9.5%** | **0.9%** | **+8.6%** | 0.0% → 0.0% | 209.6 ms |

### Metric Definitions:
- **Raw EDR (Execution Deception Rate)**: Percentage of tool failures where the raw model falsely asserted success.
- **Guarded EDR**: Deception rate after Truthify runtime verification & in-scratchpad self-correction.
- **Protection Gain (Δ)**: Percentage point reduction in deception achieved by Truthify.
- **FER (Fabrication on Empty Rate)**: Percentage of empty query returns (`[]`) where the model hallucinated fake records.