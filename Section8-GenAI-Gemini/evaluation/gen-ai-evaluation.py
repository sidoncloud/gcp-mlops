# =============================================================================
# Measuring Gemini Output Quality with the Gen AI Evaluation Service
# =============================================================================
"""Scoring model output quality with the Vertex AI Gen AI Evaluation Service.

You cannot ship what you cannot measure. The hardest part of putting an LLM
into production is not getting an answer out of it; it is knowing whether that
answer is any good, and proving the next version is better rather than just
different. Eyeballing a handful of outputs does not scale and does not survive
contact with a stakeholder asking "how do you know?".

The Gen AI Evaluation Service turns that gut feel into numbers. You hand it a
dataset of prompts and responses, choose metrics (model-based judges like
summarization quality and groundedness, or computation-based metrics), and it
returns a scored table plus aggregate summary metrics you can track over time.
This is the backbone of LLMOps: regression-test prompts, compare candidate
models before a migration, and gate releases on a quality bar.
"""

# Requires: pip install --upgrade "google-cloud-aiplatform[evaluation]"
# (the [evaluation] extra pulls in pandas and the metric dependencies)

import pandas as pd

import vertexai
from vertexai.evaluation import EvalTask, MetricPromptTemplateExamples
from google import genai

PROJECT_ID = "your-project-id"  # change this to your Google Cloud project
LOCATION = "us-central1"

# The evaluation module needs the classic vertexai init. Note this is the eval
# module, NOT the retired vertexai.generative_models SDK, so it is fine to use.
# Eval runs are executed and billed as batch prediction jobs under the hood,
# so a run over a large dataset incurs batch inference cost, not online cost.
vertexai.init(project=PROJECT_ID, location=LOCATION)

# For generating candidate responses we use the current Gen AI SDK client.
genai_client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)


# =============================================================================
# Build a Small Evaluation Dataset
# =============================================================================
# An eval dataset is just a pandas DataFrame where each row is one example. For
# a summarization task we provide the source text as the prompt and a candidate
# summary as the response. Model-based metrics read these columns by name, so
# "prompt" and "response" are the conventional column names the service expects.

eval_rows = [
    {
        "prompt": (
            "Summarize for an executive: Q3 revenue rose 12 percent to $4.2M, "
            "driven by enterprise renewals. Churn held flat at 2 percent. "
            "Support costs grew 8 percent after we hired three agents."
        ),
        "response": (
            "Q3 revenue grew 12 percent to $4.2M on strong enterprise renewals, "
            "with churn steady at 2 percent and support costs up 8 percent from "
            "added headcount."
        ),
    },
    {
        "prompt": (
            "Summarize for an executive: A datacenter outage on Tuesday took the "
            "EU region offline for 47 minutes. Root cause was a failed network "
            "switch. No data was lost and failover is now automated."
        ),
        "response": (
            "A 47-minute EU outage on Tuesday was caused by a failed switch. No "
            "data was lost, and failover has since been automated."
        ),
    },
    {
        "prompt": (
            "Summarize for an executive: The mobile app shipped dark mode and "
            "offline sync this month. Crash rate dropped 30 percent. App store "
            "rating climbed from 4.1 to 4.5 stars."
        ),
        # A deliberately weaker summary so the metrics have something to penalize
        # and the scores are not uniformly perfect.
        "response": "The app got some updates and people seem happier about it.",
    },
]

eval_dataset = pd.DataFrame(eval_rows)
print(f"Built eval dataset with {len(eval_dataset)} examples")


# =============================================================================
# Define Metrics and Run the Evaluation
# =============================================================================
# MetricPromptTemplateExamples gives you battle-tested model-based judges so you
# do not have to author judge prompts from scratch. Summarization quality rates
# how well the summary captures the source; groundedness checks that the summary
# only asserts things supported by the source (a proxy for hallucination). The
# EvalTask binds a dataset to a metric set; evaluate() scores every row and
# returns per-row results plus aggregate summary metrics.

metrics = [
    MetricPromptTemplateExamples.Pointwise.SUMMARIZATION_QUALITY,
    MetricPromptTemplateExamples.Pointwise.GROUNDEDNESS,
]

summarization_eval = EvalTask(
    dataset=eval_dataset,
    metrics=metrics,
    experiment="summary-quality-eval",
)

result = summarization_eval.evaluate()
print("Evaluation complete.")


# =============================================================================
# Aggregate Scores Across All Rows
# =============================================================================
# result.summary_metrics is a dict keyed by "<metric>/mean" and "<metric>/std".
# We print just the values we care about, formatted, so it reads cleanly on
# screen: each metric on its own line with mean and standard deviation, plus
# the row count. Higher is better for both metrics.

print("\n" + "=" * 72)
print("AGGREGATE SCORES (across all rows)")
print("=" * 72)
for metric_name in ["summarization_quality", "groundedness"]:
    mean = result.summary_metrics.get(f"{metric_name}/mean")
    std = result.summary_metrics.get(f"{metric_name}/std")
    print(f"  {metric_name:25s}  mean={mean:.2f}  std={std:.2f}")
print(f"  rows scored              {int(result.summary_metrics.get('row_count'))}")


# =============================================================================
# Per-Row Scores with the Judge's Explanation
# =============================================================================
# result.metrics_table is the full DataFrame: one row per example, columns for
# prompt, response, and per-metric score + explanation. The default print
# wraps it into unreadable chunks, so we iterate and print each row as one
# clear block: prompt, response, then each metric's score and the judge's
# plain English reasoning underneath it.

def _short(s, n=140):
    """Trim long text to one line so the block stays compact on screen."""
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[: n - 1] + "..."


print("\n" + "=" * 72)
print("PER-ROW SCORES (one block per example)")
print("=" * 72)
for i, row in result.metrics_table.iterrows():
    print(f"\n--- Row {i} ---")
    print(f"Prompt:   {_short(row['prompt'])}")
    print(f"Response: {_short(row['response'])}")
    print()
    print(f"  summarization_quality  score = {row['summarization_quality/score']:.1f} / 5")
    print(f"    why: {_short(row['summarization_quality/explanation'], 260)}")
    print(f"  groundedness           score = {row['groundedness/score']:.1f} / 1")
    print(f"    why: {_short(row['groundedness/explanation'], 260)}")


# =============================================================================
# Run the A/B Migration Eval Across Two Candidate Models
# =============================================================================
# The real payoff of eval is the A/B decision: should we move from model A to
# model B, or from prompt v1 to prompt v2? Instead of supplying a fixed
# "response" column, we pass a model to evaluate() and let the service generate
# the responses against the same prompts and the same metrics. Run it once per
# candidate, then compare the summary metrics side by side.
#
# Here we pit a cheaper, faster flash model against a stronger model on the
# same summarization task. We use a small callable wrapper so the candidate
# stays on the current google-genai SDK with no deprecated classes anywhere.

migration_dataset = pd.DataFrame(
    {"prompt": [row["prompt"] for row in eval_rows]}
)


def make_responder(model_name: str):
    """Return a callable that takes a prompt and returns a response string."""
    def respond(prompt: str) -> str:
        return genai_client.models.generate_content(
            model=model_name, contents=prompt
        ).text
    return respond


candidates = {
    "gemini-2.5-flash": make_responder("gemini-2.5-flash"),
    "gemini-2.5-pro": make_responder("gemini-2.5-pro"),
}

comparison = {}
for name, responder in candidates.items():
    task = EvalTask(
        dataset=migration_dataset,
        metrics=metrics,
        experiment="model-migration-eval",
    )
    # Passing model= makes the service call the model to produce responses,
    # then score them. This is what makes the comparison apples to apples.
    candidate_result = task.evaluate(model=responder)
    comparison[name] = candidate_result.summary_metrics

print(f"\nScored {len(comparison)} candidate models on the same prompts.")


# =============================================================================
# Compare the Two Models Side by Side
# =============================================================================
# Build a small DataFrame with one row per model and one column per metric mean
# / std, then print it as a clean side-by-side. Higher is better on both means.

rows = []
for name, summary in comparison.items():
    rows.append({
        "model": name,
        "quality_mean": summary.get("summarization_quality/mean"),
        "quality_std": summary.get("summarization_quality/std"),
        "groundedness_mean": summary.get("groundedness/mean"),
        "groundedness_std": summary.get("groundedness/std"),
    })
compare_df = pd.DataFrame(rows).set_index("model")

print("\n" + "=" * 72)
print("MODEL COMPARISON  (means and standard deviations, higher is better)")
print("=" * 72)
print(compare_df.round(3).to_string())


# =============================================================================
# Compute Deltas and Apply the Decision Rule
# =============================================================================
# Deltas (pro minus flash). Positive on quality means pro is better. Positive
# on groundedness means pro is better. Negative numbers favor flash.
#
# Decision rule: prefer the cheaper model (flash) unless pro wins on BOTH
# axes by a meaningful margin. Groundedness is the hallucination canary, so
# any groundedness regression is disqualifying regardless of quality.

flash = compare_df.loc["gemini-2.5-flash"]
pro = compare_df.loc["gemini-2.5-pro"]
quality_delta = pro["quality_mean"] - flash["quality_mean"]
ground_delta = pro["groundedness_mean"] - flash["groundedness_mean"]

print("\nDeltas (pro - flash):")
print(f"  quality       {quality_delta:+.2f}   (positive = pro wins on quality)")
print(f"  groundedness  {ground_delta:+.2f}   (positive = pro wins on groundedness)")

print("\nDecision:")
if ground_delta < 0:
    print("  -> promote gemini-2.5-flash")
    print("     pro has lower groundedness; a bigger model that hallucinates")
    print("     more is not a safe production migration even if quality is up.")
elif quality_delta > 0.5 and ground_delta >= 0:
    print("  -> promote gemini-2.5-pro")
    print("     pro wins on both quality and groundedness, and the quality gap")
    print("     is large enough to justify the higher per-token cost.")
else:
    print("  -> stay on gemini-2.5-flash")
    print("     gaps are within noise; the cheaper, faster model wins on TCO.")
