"""
06_build_request.py

Reads the unseen orders CSV from GCS (orders the model was NOT trained on),
applies the same categorical encoding as training, and writes a Vertex AI
prediction request file `request.json` that can be posted to the deployed
endpoint with curl.

Run:
    python 06_build_request.py --bucket $BUCKET_NAME --rows 5
"""
import argparse
import json

import pandas as pd


CATEGORY_MAP = {
    "Clothing": 0, "Electronics": 1, "Home": 2, "Sports": 3,
    "Books": 4, "Beauty": 5, "Toys": 6,
}
PAYMENT_MAP = {
    "credit_card": 0, "paypal": 1, "debit_card": 2, "apple_pay": 3, "gift_card": 4,
}

FEATURE_ORDER = [
    "order_total",
    "num_items",
    "item_price",
    "discount_applied_percent",
    "shipping_days",
    "category_code",
    "product_avg_rating",
    "customer_past_order_count",
    "customer_past_return_rate",
    "customer_tenure_days",
    "payment_code",
    "is_first_purchase",
    "used_size_guide",
    "promo_used",
    "weekend_order",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True, help="GCS bucket holding new/orders_new.csv")
    parser.add_argument("--rows", type=int, default=5, help="How many rows to include in the request")
    parser.add_argument("--output", default="request.json")
    args = parser.parse_args()

    uri = f"gs://{args.bucket}/new/orders_new.csv"
    print(f"Reading unseen orders from {uri}")
    df = pd.read_csv(uri)
    df = df.head(args.rows).copy()
    print(f"Loaded {len(df)} unseen order(s)")

    df["category_code"] = df["product_category"].map(CATEGORY_MAP)
    df["payment_code"] = df["payment_method"].map(PAYMENT_MAP)

    instances = df[FEATURE_ORDER].astype(float).values.tolist()
    payload = {"instances": instances}

    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote prediction request with {len(instances)} instance(s) to {args.output}")
    print("Order IDs included:", df["order_id"].tolist())


if __name__ == "__main__":
    main()
