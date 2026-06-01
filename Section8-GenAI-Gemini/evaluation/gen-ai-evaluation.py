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

Note on environment: this lab targets the Gen AI Eval SDK shipped in recent
google-cloud-aiplatform releases. The eval module was NOT importable in the
build environment (google-cloud-aiplatform 1.40.0 predates it, and that build
also had a protobuf / proto-plus mismatch), so the code here follows the
documented SDK surface. Install the eval extra before running:
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

print("Summary metrics:")
print(result.summary_metrics)
print("\nPer-example scores:")
print(result.metrics_table)


# =============================================================================
# Compare Two Models to Make a Migration Decision
# =============================================================================
# The real payoff of eval is the A/B decision: should we move from model A to
# model B, or from prompt v1 to prompt v2? Instead of supplying a fixed
# "response" column, we pass a model to evaluate() and let the service generate
# the responses against the same prompts and the same metrics. Run it once per
# candidate, then compare the summary metrics side by side.
#
# Here we pit a cheaper, faster flash model against a stronger model on the same
# summarization task. Whichever wins on summarization quality without losing
# groundedness is the one you promote.

migration_dataset = pd.DataFrame(
    {"prompt": [row["prompt"] for row in eval_rows]}
)


# EvalTask accepts a callable as the model: a function that takes the prompt
# string and returns the response string. We use the current Gen AI SDK inside
# it, so no part of this lab depends on the retired generative_models classes.
def make_responder(model_name: str):
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

print("\nModel comparison (summary metrics):")
for name, summary in comparison.items():
    print(f"\n{name}:")
    print(summary)

# Decision rule in practice: promote the model with the higher mean
# summarization_quality, provided its groundedness does not drop below your
# release bar. If flash is within a small margin of pro, the cheaper model
# usually wins on total cost of ownership.
