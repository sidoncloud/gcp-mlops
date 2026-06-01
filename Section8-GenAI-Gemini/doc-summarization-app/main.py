"""
Document Summarization API - Section 8 (Gemini on Cloud Run)

A small Flask service that summarizes documents with Gemini 2.5 Flash via the
current Google Gen AI SDK (google-genai). Two ways in:
  POST /summarize       -> summarize raw text in the request body
  POST /summarize_gcs   -> summarize a PDF stored in Cloud Storage
  GET  /health          -> liveness check for Cloud Run
"""

import os

from flask import Flask, jsonify, request
from google import genai
from google.genai import types

app = Flask(__name__)

# On Cloud Run we route through Vertex AI using the service identity. Project
# and location come from env vars set at deploy time (see deploy-commands.sh).
client = genai.Client(
    vertexai=True,
    project=os.environ.get("GOOGLE_CLOUD_PROJECT"),
    location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
)

MODEL = "gemini-2.5-flash"

SYSTEM_INSTRUCTION = (
    "You are a precise summarization assistant. Capture the key points "
    "faithfully and never invent facts that are not in the source."
)


@app.get("/health")
def health():
    return jsonify(status="ok", model=MODEL)


@app.post("/summarize")
def summarize():
    body = request.get_json(silent=True) or {}
    text = (body.get("text") or "").strip()
    if not text:
        return jsonify(error="Field 'text' is required."), 400

    response = client.models.generate_content(
        model=MODEL,
        contents=f"Summarize the following document concisely:\n\n{text}",
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.2,
        ),
    )

    return jsonify(summary=response.text)


@app.post("/summarize_gcs")
def summarize_gcs():
    body = request.get_json(silent=True) or {}
    gcs_uri = (body.get("gcs_uri") or "").strip()
    if not gcs_uri.startswith("gs://"):
        return jsonify(error="Field 'gcs_uri' must be a gs:// URI."), 400

    # Gemini reads the PDF directly from GCS as a multimodal Part, so we never
    # download or parse the file ourselves.
    pdf_part = types.Part.from_uri(
        file_uri=gcs_uri,
        mime_type="application/pdf",
    )

    response = client.models.generate_content(
        model=MODEL,
        contents=[pdf_part, "Summarize this document in 5 bullet points."],
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.2,
        ),
    )

    return jsonify(summary=response.text, source=gcs_uri)


if __name__ == "__main__":
    # Local dev only. In production gunicorn serves the app (see Dockerfile).
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
