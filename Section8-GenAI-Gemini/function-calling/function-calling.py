"""
Function Calling (Tools) with Google Gemini on Vertex AI

This lab shows how to let Gemini call real external code: a customer-facing
support assistant that answers questions using live business data (order status
and shipping estimates) instead of guessing.

We cover two modes:
  1. Automatic function calling - the SDK inspects your Python functions, calls
     them when the model asks, and returns the final natural-language answer.
  2. Manual function calling - you inspect the model's function_call request,
     run the function yourself, and feed the result back. This is what is
     happening under the hood, and it is what you use when functions must run
     on a separate service, need approval, or hit a real production API.

Verified against google-genai v1.14.0.
"""

# =============================================================================
# Cell 1: Imports and Client Setup
# =============================================================================
from google import genai
from google.genai import types

PROJECT_ID = "your-project-id"  # change this to your Google Cloud project ID
LOCATION = "us-central1"
MODEL = "gemini-2.5-flash"

client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)


# =============================================================================
# Cell 2: Define the Tools as Plain Python Functions
# =============================================================================
# These functions stand in for real systems: an order database and a shipping
# API. The type hints and docstrings are not optional decoration. The SDK reads
# them to build the tool schema it hands to the model, so be explicit and clear.

def get_order_status(order_id: str) -> dict:
    """Look up the current status of a customer order by its order ID.

    Args:
        order_id: The customer's order identifier, for example 'A1234'.

    Returns:
        A dictionary with the order's current status and last known location.
    """
    # In production this would query your orders database or an internal API.
    mock_orders = {
        "A1234": {"status": "in_transit", "current_location": "Pune hub", "carrier": "BlueDart"},
        "B5678": {"status": "delivered", "current_location": "Delhi", "carrier": "Delhivery"},
    }
    return mock_orders.get(
        order_id,
        {"status": "not_found", "message": f"No order found with ID {order_id}."},
    )


def get_shipping_estimate(destination_city: str) -> dict:
    """Estimate remaining delivery time to a destination city.

    Args:
        destination_city: The city the order is being shipped to, for example 'Mumbai'.

    Returns:
        A dictionary with the estimated days remaining and a delivery window.
    """
    # In production this would call your carrier's live shipping API.
    mock_estimates = {
        "mumbai": {"estimated_days": 2, "delivery_window": "May 31 to Jun 1"},
        "bangalore": {"estimated_days": 3, "delivery_window": "Jun 1 to Jun 2"},
        "chennai": {"estimated_days": 4, "delivery_window": "Jun 2 to Jun 3"},
    }
    return mock_estimates.get(
        destination_city.lower(),
        {"estimated_days": 5, "delivery_window": "within a week", "note": "default estimate"},
    )


# =============================================================================
# Cell 3: Automatic Function Calling
# =============================================================================
# Pass the Python functions straight into tools. The SDK builds the schema,
# detects when Gemini wants a function, runs it, sends the result back, and
# loops until the model produces a final answer. We never touch the plumbing.

question = "Where is my order A1234 and when will it reach Mumbai?"

auto_response = client.models.generate_content(
    model=MODEL,
    contents=question,
    config=types.GenerateContentConfig(
        tools=[get_order_status, get_shipping_estimate],
    ),
)

# The SDK already called get_order_status('A1234') and get_shipping_estimate('Mumbai')
# for us, so this is the finished, customer-ready answer.
print("=== Automatic function calling ===")
print(auto_response.text)


# =============================================================================
# Cell 4: Manual Function Calling Walkthrough
# =============================================================================
# Same question, but we turn automatic execution OFF so we can see and control
# each step. This is the pattern you reach for when the function lives behind a
# real API, needs auth, or should be audited before it runs.

# Describe the tool to the model with an explicit function declaration. The
# 'parameters' schema mirrors the function signature using OpenAPI-style types.
order_status_declaration = types.FunctionDeclaration(
    name="get_order_status",
    description="Look up the current status of a customer order by its order ID.",
    parameters=types.Schema(
        type="OBJECT",
        properties={
            "order_id": types.Schema(
                type="STRING",
                description="The customer's order identifier, for example 'A1234'.",
            ),
        },
        required=["order_id"],
    ),
)

support_tool = types.Tool(function_declarations=[order_status_declaration])

# disable=True stops the SDK from auto-running the function so we can inspect
# the request ourselves. (Field verified: AutomaticFunctionCallingConfig.disable)
manual_config = types.GenerateContentConfig(
    tools=[support_tool],
    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
)

# First turn: the model responds with a function_call instead of text.
first_turn = client.models.generate_content(
    model=MODEL,
    contents="What is the status of order A1234?",
    config=manual_config,
)

function_call = first_turn.candidates[0].content.parts[0].function_call
print("\n=== Manual function calling ===")
print("Model requested:", function_call.name, "with args", dict(function_call.args))

# We execute the function ourselves, exactly as the model asked.
result = get_order_status(**function_call.args)
print("We ran the function and got:", result)

# Send the result back as a function_response part. We replay the original
# question and the model's function_call, then append our result so the model
# has the full context to write a final answer.
# (Signature verified: types.Part.from_function_response(*, name, response))
follow_up = client.models.generate_content(
    model=MODEL,
    contents=[
        types.Content(role="user", parts=[types.Part(text="What is the status of order A1234?")]),
        first_turn.candidates[0].content,
        types.Content(
            role="user",
            parts=[types.Part.from_function_response(name=function_call.name, response=result)],
        ),
    ],
    config=manual_config,
)

print("Final answer:", follow_up.text)


# =============================================================================
# Cell 5: When to Use Function Calling in Production
# =============================================================================
# Reach for function calling whenever the model needs facts or actions it cannot
# invent on its own:
#   - Live systems and databases (order status, inventory, account balances).
#   - External APIs (shipping, payments, weather, search).
#   - Side effects (create a ticket, send an email, trigger a workflow).
# Start with automatic mode for simple in-process helpers. Switch to manual mode
# when the function runs on another service, requires authentication, or must be
# approved or logged before it executes.
