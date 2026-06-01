"""
Section 8 - Your First Gemini Calls on Vertex AI
Uses the current Google Gen AI SDK (google-genai), not the retired
vertexai.generative_models SDK.

Prerequisites:
    pip install google-genai
    gcloud auth application-default login
"""

# =============================================================================
# Setup and Client Initialization
# =============================================================================
from google import genai
from google.genai import types

# The Gen AI SDK is client-based. Setting vertexai=True routes calls through
# Vertex AI using your GCP project, instead of the public Gemini Developer API.
PROJECT_ID = "your-project-id"  # <-- CHANGE THIS
LOCATION = "us-central1"

client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

# We pin every lab to the GA 2.5 Flash model so nothing breaks mid-course.
# Gemini 3.x is the current frontier if you want to swap it in later.
MODEL = "gemini-2.5-flash"

print("Gen AI client ready, using", MODEL)

# =============================================================================
# Your First Text Generation
# =============================================================================
response = client.models.generate_content(
    model=MODEL,
    contents="Explain what an LLM is to a backend engineer in three sentences.",
)

print(response.text)

# =============================================================================
# Controlling the Output with a Generation Config
# =============================================================================
# temperature controls randomness, max_output_tokens caps the response length.
response = client.models.generate_content(
    model=MODEL,
    contents="Write a one-line tagline for a bike-sharing startup.",
    config=types.GenerateContentConfig(
        temperature=0.9,
        max_output_tokens=64,
    ),
)

print(response.text)

# =============================================================================
# System Instructions - Giving the Model a Role
# =============================================================================
# System instructions set persistent behavior for every turn, separate from
# the user prompt. This replaces the old PaLM "context" parameter.
response = client.models.generate_content(
    model=MODEL,
    contents="A customer says their order never arrived. Draft a reply.",
    config=types.GenerateContentConfig(
        system_instruction=(
            "You are a calm, concise customer support agent for an online "
            "retailer. Always apologize once, then give a clear next step. "
            "Never promise a refund without manager approval."
        ),
        temperature=0.3,
    ),
)

print(response.text)

# =============================================================================
# Multi-Turn Chat
# =============================================================================
# The chat object keeps conversation history for you across turns.
chat = client.chats.create(model=MODEL)

print(chat.send_message("I want to start a sneaker reselling business.").text)
print("---")
print(chat.send_message("What are the three biggest risks?").text)
print("---")
print(chat.send_message("Summarize your advice as a checklist.").text)

# Inspect the stored history
print("\n=== Chat history ===")
for message in chat.get_history():
    print(f"[{message.role}] {message.parts[0].text[:80]}...")

# =============================================================================
# Multimodal Input - Sending Gemini an Image
# =============================================================================
# Gemini is natively multimodal. Pass an image Part alongside a text prompt in
# the same call. Here we point at a public image in Google Cloud Storage.
image_part = types.Part.from_uri(
    file_uri="gs://cloud-samples-data/generative-ai/image/scones.jpg",
    mime_type="image/jpeg",
)

response = client.models.generate_content(
    model=MODEL,
    contents=[
        image_part,
        "Describe this image, then list every food item you can see.",
    ],
)

print(response.text)

# =============================================================================
# Streaming the Response
# =============================================================================
# For chat-style UIs you stream tokens as they are generated instead of
# waiting for the full response.
print("Streaming response:\n")
for chunk in client.models.generate_content_stream(
    model=MODEL,
    contents="List five real-world use cases for LLMs in an e-commerce company.",
):
    print(chunk.text, end="")
print()

# =============================================================================
# Counting Tokens Before You Send
# =============================================================================
# Token counts drive cost and context limits. Check them up front.
prompt = "Summarize the entire history of cloud computing."
token_info = client.models.count_tokens(model=MODEL, contents=prompt)
print(f"This prompt is {token_info.total_tokens} tokens.")
