# =============================================================================
# Reliable Structured JSON Output from Gemini
# =============================================================================
"""Structured data extraction with Google Gemini.

In real data pipelines we constantly need to turn messy free text (invoices,
resumes, support tickets, scraped pages) into clean, typed records that a
downstream system can store and query. Asking a model to "return JSON" in a
plain text prompt is fragile: the output drifts, gets wrapped in markdown,
or quietly changes shape.

This lab shows the dependable approach. We define the target shape as Pydantic
models and hand them to Gemini as a response schema. Gemini is then constrained
to emit JSON that matches the schema, and the SDK hands us back a parsed,
typed object we can use directly in code.
"""

from enum import Enum

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

PROJECT_ID = "your-project-id"  # change this to your Google Cloud project
LOCATION = "us-central1"
MODEL = "gemini-2.5-flash"

client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)


# =============================================================================
# The Naive Approach and Why It Fails
# =============================================================================
# The tempting first move is to just ask the model for JSON in plain text.
# The trouble is that nothing constrains the output: the model often wraps the
# JSON in a ```json markdown fence, adds a chatty preamble like "Here is the
# JSON you asked for", or invents field names. So json.loads() on response.text
# blows up, and you end up writing brittle regex to strip fences. That cleanup
# code breaks the moment the model phrases things slightly differently. This is
# exactly the fragility that schema-constrained structured output removes.

naive_prompt = """Extract the vendor, invoice number, and total from this text
and return it as JSON:

Invoice from Bluewave Office Supplies, #INV-2025-0042, total due $1,284.50."""

naive_response = client.models.generate_content(model=MODEL, contents=naive_prompt)

print("Naive raw text output:")
print(naive_response.text)
# Notice the output is a string, frequently fenced or prefixed, not a clean
# object you can trust. We have to guess at its format every single time.


# =============================================================================
# Define the Pydantic Models for an Invoice
# =============================================================================
class Currency(str, Enum):
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"


class LineItem(BaseModel):
    description: str = Field(description="What the line item is for")
    quantity: int = Field(description="Number of units purchased")
    unit_price: float = Field(description="Price for a single unit")


class Invoice(BaseModel):
    vendor_name: str = Field(description="Company that issued the invoice")
    invoice_number: str = Field(description="Unique invoice identifier")
    invoice_date: str = Field(description="Invoice date in YYYY-MM-DD form")
    currency: Currency = Field(description="Currency of the amounts")
    line_items: list[LineItem]
    total_amount: float = Field(description="Final total owed on the invoice")


# =============================================================================
# Run the Invoice Extraction
# =============================================================================
invoice_blob = """
BLUEWAVE OFFICE SUPPLIES
Invoice #INV-2025-0042   Date: March 14, 2025

Thanks for your order! Here is the breakdown:
- 4x ergonomic office chairs at 189.00 each
- 2 standing desks, 425.50 a piece
- A box of assorted pens (qty 1) for 23.50

Everything billed in US dollars. Amount due: 1,630.50
"""

invoice_response = client.models.generate_content(
    model=MODEL,
    contents=f"Extract the invoice details from the following text:\n{invoice_blob}",
    config=types.GenerateContentConfig(
        temperature=0,
        system_instruction="You are a precise invoice data extraction service.",
        response_mime_type="application/json",
        response_schema=Invoice,
    ),
)

# response.parsed is already an instance of Invoice, not a string to clean up.
invoice: Invoice = invoice_response.parsed

print(f"\nVendor: {invoice.vendor_name}")
print(f"Invoice number: {invoice.invoice_number}")
print(f"Date: {invoice.invoice_date}")
print(f"Currency: {invoice.currency.value}")
print(f"Total: {invoice.total_amount}")
print("Line items:")
for item in invoice.line_items:
    print(f"  {item.quantity} x {item.description} @ {item.unit_price}")


# =============================================================================
# A Second Example: Extracting a Resume
# =============================================================================
# The same pattern works for any domain. Define the shape, pass the schema,
# get a typed object back. Here we pull a candidate record out of a resume.
class Candidate(BaseModel):
    name: str = Field(description="Full name of the candidate")
    years_experience: int = Field(description="Total years of work experience")
    skills: list[str] = Field(description="Notable technical skills")
    most_recent_role: str = Field(description="Job title of the latest position")


resume_blob = """
Maria Gonzalez

Over the last 8 years I have worked across data and machine learning teams.
Most recently I have been a Senior MLOps Engineer at Northstar Analytics, where
I owned the model deployment platform. Comfortable with Python, Kubernetes,
Terraform, BigQuery, and building CI/CD for ML.
"""

resume_response = client.models.generate_content(
    model=MODEL,
    contents=f"Extract the candidate profile from this resume:\n{resume_blob}",
    config=types.GenerateContentConfig(
        temperature=0,
        response_mime_type="application/json",
        response_schema=Candidate,
    ),
)

candidate: Candidate = resume_response.parsed

print(f"\nCandidate: {candidate.name}")
print(f"Experience: {candidate.years_experience} years")
print(f"Most recent role: {candidate.most_recent_role}")
print(f"Skills: {', '.join(candidate.skills)}")


# =============================================================================
# Why This Matters for Pipelines
# =============================================================================
# Because we have a real typed object, we can run business logic on it with
# confidence instead of parsing strings. Below we validate the line item math
# and assemble a database-ready record.
computed_total = sum(item.quantity * item.unit_price for item in invoice.line_items)
print(f"\nComputed line item total: {computed_total}")
print(f"Reported invoice total: {invoice.total_amount}")

# Flag a mismatch the way a real ingestion job would before storing the record.
if abs(computed_total - invoice.total_amount) > 0.01:
    print("WARNING: line items do not reconcile with the stated total")
else:
    print("Totals reconcile.")

# A clean dict ready to insert into a database or queue for downstream work.
invoice_record = {
    "vendor": invoice.vendor_name,
    "invoice_number": invoice.invoice_number,
    "invoice_date": invoice.invoice_date,
    "currency": invoice.currency.value,
    "total_amount": invoice.total_amount,
    "line_item_count": len(invoice.line_items),
}
print(f"\nDatabase-ready record: {invoice_record}")
