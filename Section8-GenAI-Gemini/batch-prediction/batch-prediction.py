# =============================================================================
# Batch Prediction at Scale with Gemini on Vertex AI
# =============================================================================
"""Scoring large volumes of inputs offline with Gemini batch jobs.

Online generate_content calls are perfect for interactive, low-latency work:
a chatbot turn, a single form submission, one document on demand. But a lot of
real MLOps work is bulk scoring. You have a million support tickets to classify
overnight, ten thousand product reviews to extract sentiment from, or a backlog
of documents to tag before a migration. Firing those through the online API one
at a time is slow, hits rate limits, and costs more than it should.

Batch prediction is the answer. You hand Vertex AI a file of requests sitting
in Cloud Storage (or a BigQuery table), it works through them asynchronously on
managed infrastructure, and it writes the results back out. Batch is roughly
50 percent cheaper per token than online inference and is the right tool for
large, latency-tolerant offline workloads.

This lab builds a batch job end to end: we construct a JSONL request file,
upload it to GCS, submit the job, poll until it finishes, and read back the
results.
"""

import json
import time

from google import genai
from google.genai import types
from google.cloud import storage

PROJECT_ID = "your-project-id"  # change this to your Google Cloud project
LOCATION = "us-central1"
MODEL = "gemini-2.5-flash"

# change these to a bucket and prefix you own; the input JSONL is written here
# and Vertex AI writes the prediction output under the same bucket
BUCKET_NAME = "your-batch-bucket"
INPUT_BLOB = "batch/input/requests.jsonl"
OUTPUT_PREFIX = "batch/output"

client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)


# =============================================================================
# When Batch Beats Online Inference
# =============================================================================
# Reach for batch prediction when all three of these are true:
#   1. You have many inputs (hundreds to millions), not a single request.
#   2. You do not need the answer this second; minutes to hours is fine.
#   3. You care about throughput and cost more than latency.
# Classic fits: nightly classification of a ticket queue, bulk entity
# extraction over a document corpus, backfilling labels on historical data,
# generating embeddings-style features for a whole dataset.
#
# Online inference is the opposite trade: you pay a premium for immediacy and
# you process one request at a time. If a human is waiting on the response,
# use online. If a pipeline is processing a backlog, use batch.


# =============================================================================
# Build the JSONL Request File
# =============================================================================
# A batch input file is JSON Lines: one self-contained request per line. Each
# line uses the same "request" shape you would pass to generate_content, so the
# model field, the contents, and any generation config all live inside it.
# Here we run a simple classification task over a list of support messages,
# asking Gemini to bucket each one into a fixed set of categories.

support_messages = [
    "My invoice charged me twice for the same month, please refund the extra.",
    "How do I export all of my data before I cancel my account?",
    "The dashboard has been throwing a 500 error since this morning.",
    "Can you walk me through upgrading from the Starter to the Pro plan?",
    "I never received the password reset email even after three tries.",
]

CATEGORIES = ["billing", "account", "bug", "sales", "auth"]

system_instruction = (
    "You are a support triage assistant. Classify the user message into exactly "
    f"one of these categories: {', '.join(CATEGORIES)}. "
    "Reply with only the single category word."
)


def build_request(message: str) -> dict:
    """Return one batch request row in the structure Vertex AI expects.

    The "request" key holds a standard GenerateContent payload: a contents list
    of role/parts plus a generationConfig. Keeping each row fully self-contained
    is what lets the service fan the work out across many workers.
    """
    return {
        "request": {
            "contents": [
                {"role": "user", "parts": [{"text": message}]},
            ],
            "systemInstruction": {"parts": [{"text": system_instruction}]},
            "generationConfig": {"temperature": 0.0},
        }
    }


local_input_path = "requests.jsonl"
with open(local_input_path, "w") as f:
    for message in support_messages:
        f.write(json.dumps(build_request(message)) + "\n")

print(f"Wrote {len(support_messages)} requests to {local_input_path}")


# =============================================================================
# Upload the Request File to Cloud Storage
# =============================================================================
# Batch jobs read their input from GCS (or BigQuery), never from your laptop.
# We upload the JSONL with the google-cloud-storage client. If you prefer the
# command line you can instead run:
#   gsutil cp requests.jsonl gs://your-batch-bucket/batch/input/requests.jsonl

storage_client = storage.Client(project=PROJECT_ID)
bucket = storage_client.bucket(BUCKET_NAME)
bucket.blob(INPUT_BLOB).upload_from_filename(local_input_path)

input_uri = f"gs://{BUCKET_NAME}/{INPUT_BLOB}"
output_uri = f"gs://{BUCKET_NAME}/{OUTPUT_PREFIX}"
print(f"Uploaded input to {input_uri}")


# =============================================================================
# Submit the Batch Job
# =============================================================================
# create() takes the model, the GCS source URI (src), and a config describing
# where to write results. The destination format is "jsonl" to match GCS
# output; for a BigQuery destination you would set bigquery_uri instead and a
# "bigquery" format. The call returns immediately with a BatchJob handle; the
# work happens asynchronously on Vertex AI infrastructure.

batch_job = client.batches.create(
    model=MODEL,
    src=input_uri,
    config=types.CreateBatchJobConfig(
        display_name="support-triage-batch",
        dest=types.BatchJobDestination(
            format="jsonl",
            gcs_uri=output_uri,
        ),
    ),
)

print(f"Submitted batch job: {batch_job.name}")
print(f"Initial state: {batch_job.state}")


# =============================================================================
# Poll Until the Job Finishes
# =============================================================================
# A batch job moves through queued and running states before it lands in a
# terminal state. We poll batches.get() on an interval until we hit one of the
# finished states. In production you would wire this to an orchestrator (Cloud
# Composer, Workflows) rather than blocking a script, but the polling logic is
# the same idea.

TERMINAL_STATES = {
    types.JobState.JOB_STATE_SUCCEEDED,
    types.JobState.JOB_STATE_FAILED,
    types.JobState.JOB_STATE_CANCELLED,
    types.JobState.JOB_STATE_PARTIALLY_SUCCEEDED,
    types.JobState.JOB_STATE_EXPIRED,
}

job = client.batches.get(name=batch_job.name)
while job.state not in TERMINAL_STATES:
    print(f"State: {job.state} ... waiting")
    time.sleep(30)
    job = client.batches.get(name=batch_job.name)

print(f"Final state: {job.state}")
if job.state != types.JobState.JOB_STATE_SUCCEEDED:
    # Surface the error so a failed run does not look like an empty success.
    print(f"Job did not fully succeed. Error detail: {job.error}")


# =============================================================================
# Read and Print the Results
# =============================================================================
# Vertex AI writes a predictions file under the output prefix. We list the
# objects at that prefix, find the predictions JSONL, and parse each line. Every
# output row pairs the original request with the model's response, so order and
# correlation are preserved for joining back to your source data.

result_uri = job.dest.gcs_uri if job.dest else output_uri
result_prefix = result_uri.replace(f"gs://{BUCKET_NAME}/", "")

print(f"Reading results from {result_uri}")
for blob in storage_client.list_blobs(BUCKET_NAME, prefix=result_prefix):
    if not blob.name.endswith(".jsonl"):
        continue
    for line in blob.download_as_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        # The response shape mirrors generate_content: candidates -> content
        # -> parts -> text. We pull the predicted category out of the first
        # candidate. Defensive .get() chaining keeps a single malformed row
        # from killing the whole read.
        candidates = record.get("response", {}).get("candidates", [])
        prediction = ""
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            prediction = parts[0].get("text", "").strip() if parts else ""
        print(f"-> {prediction}")
