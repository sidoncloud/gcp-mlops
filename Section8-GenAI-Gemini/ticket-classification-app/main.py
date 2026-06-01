"""
Support Ticket Triage API - Section 8 (Gemini on Cloud Run)

A Flask service that classifies an incoming support message into a category,
assigns a priority, and explains why. It uses Gemini structured output
(a Pydantic response schema) so the model returns validated JSON, not free text.

  POST /classify  -> {"message": "..."} returns category, priority, reason
  GET  /health    -> liveness check for Cloud Run
"""

import os
from enum import Enum

from flask import Flask, jsonify, request
from google import genai
from google.genai import types
from pydantic import BaseModel

app = Flask(__name__)

# On Cloud Run we route through Vertex AI using the service identity. Project
# and location come from env vars set at deploy time (see deploy-commands.sh).
client = genai.Client(
    vertexai=True,
    project=os.environ.get("GOOGLE_CLOUD_PROJECT"),
    location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
)

MODEL = "gemini-2.5-flash"


# Enums constrain the model to a fixed set of values, so downstream systems
# (routing, dashboards) never see an unexpected label.
class Category(str, Enum):
    BILLING = "Billing"
    TECHNICAL = "Technical"
    ACCOUNT = "Account"
    SHIPPING = "Shipping"
    OTHER = "Other"


class Priority(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


class Triage(BaseModel):
    category: Category
    priority: Priority
    reason: str  # one-line justification


@app.get("/health")
def health():
    return jsonify(status="ok", model=MODEL)


@app.post("/classify")
def classify():
    body = request.get_json(silent=True) or {}
    message = (body.get("message") or "").strip()
    if not message:
        return jsonify(error="Field 'message' is required."), 400

    response = client.models.generate_content(
        model=MODEL,
        contents=f"Triage this support ticket:\n\n{message}",
        config=types.GenerateContentConfig(
            system_instruction=(
                "You triage customer support tickets. Choose the best category "
                "and a priority, and give a single short sentence explaining "
                "your choice."
            ),
            temperature=0.0,
            response_mime_type="application/json",
            response_schema=Triage,
        ),
    )

    # response.parsed is a validated Triage instance built from the model's JSON.
    triage: Triage = response.parsed
    return jsonify(
        category=triage.category.value,
        priority=triage.priority.value,
        reason=triage.reason,
    )


if __name__ == "__main__":
    # Local dev only. In production gunicorn serves the app (see Dockerfile).
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
