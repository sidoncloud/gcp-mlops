"""
06_build_request.py

Reads the unseen cart sessions CSV from GCS (the data the model was NOT trained
on), applies the same categorical encoding as training, and writes a
Vertex AI prediction request file `request.json` that can be posted to the
deployed endpoint with curl.

Run:
    python 06_build_request.py --bucket $BUCKET_NAME --rows 5
"""
import argparse
import json

import pandas as pd


DEVICE_MAP = {"mobile": 0, "desktop": 1, "tablet": 2}
TRAFFIC_MAP = {"organic": 0, "paid": 1, "direct": 2, "email": 3}

FEATURE_ORDER = [
    "session_duration_seconds",
    "pages_viewed",
    "items_in_cart",
    "cart_total_value",
    "unique_categories_viewed",
    "has_discount_code",
    "is_returning_customer",
    "device_code",
    "traffic_code",
    "hour_of_day",
    "day_of_week",
    "previous_purchases_count",
    "avg_time_per_page_seconds",
    "added_to_wishlist",
    "used_search",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True, help="GCS bucket holding new/cart_sessions_new.csv")
    parser.add_argument("--rows", type=int, default=5, help="How many rows to include in the request")
    parser.add_argument("--output", default="request.json")
    args = parser.parse_args()

    uri = f"gs://{args.bucket}/new/cart_sessions_new.csv"
    print(f"Reading unseen sessions from {uri}")
    df = pd.read_csv(uri)
    df = df.head(args.rows).copy()
    print(f"Loaded {len(df)} unseen session(s)")

    df["device_code"] = df["device_type"].map(DEVICE_MAP)
    df["traffic_code"] = df["traffic_source"].map(TRAFFIC_MAP)

    instances = df[FEATURE_ORDER].astype(float).values.tolist()
    payload = {"instances": instances}

    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote prediction request with {len(instances)} instance(s) to {args.output}")
    print("Session IDs included:", df["session_id"].tolist())


if __name__ == "__main__":
    main()
