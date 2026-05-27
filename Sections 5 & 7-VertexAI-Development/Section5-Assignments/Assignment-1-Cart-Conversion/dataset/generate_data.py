"""
Dataset generator for the Cart Conversion Scoring assignment.

Outputs two CSV files:
  - cart_sessions_train.csv  (20,000 raw rows with 'converted' target)
                             Read by model-training-code.py for training.
  - cart_sessions_new.csv    (100 PRE-ENCODED rows, no target, no session_id)
                             Column order matches the model's expected feature
                             order exactly, so the orchestrator script and curl
                             can submit rows directly to the deployed endpoint
                             without any client-side preprocessing.
"""
import csv
import random
import pandas as pd
from pathlib import Path

random.seed(42)
OUTPUT_DIR = Path(__file__).parent

DEVICES = ["mobile", "desktop", "tablet"]
DEVICE_WEIGHTS = [0.55, 0.35, 0.10]

TRAFFIC_SOURCES = ["organic", "paid", "direct", "email"]
TRAFFIC_WEIGHTS = [0.40, 0.25, 0.20, 0.15]


def generate_session(session_id: str, include_target: bool = True) -> dict:
    is_returning = 1 if random.random() < 0.40 else 0
    previous_purchases = random.randint(1, 50) if is_returning else 0
    device = random.choices(DEVICES, weights=DEVICE_WEIGHTS, k=1)[0]
    traffic = random.choices(TRAFFIC_SOURCES, weights=TRAFFIC_WEIGHTS, k=1)[0]
    items_in_cart = random.choices([1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                   weights=[40, 25, 15, 8, 5, 3, 2, 1, 0.5, 0.5], k=1)[0]
    cart_total_value = round(items_in_cart * random.uniform(15.0, 180.0), 2)
    pages_viewed = random.randint(1, 30)
    session_duration_seconds = random.randint(20, 1800)
    unique_categories = min(random.randint(1, 6), items_in_cart + 2)
    has_discount_code = 1 if random.random() < 0.25 else 0
    added_to_wishlist = 1 if random.random() < 0.22 else 0
    used_search = 1 if random.random() < 0.55 else 0
    hour_of_day = random.randint(0, 23)
    day_of_week = random.randint(0, 6)
    avg_time_per_page = round(session_duration_seconds / max(pages_viewed, 1), 2)

    session = {
        "session_id": session_id,
        "session_duration_seconds": session_duration_seconds,
        "pages_viewed": pages_viewed,
        "items_in_cart": items_in_cart,
        "cart_total_value": cart_total_value,
        "unique_categories_viewed": unique_categories,
        "has_discount_code": has_discount_code,
        "is_returning_customer": is_returning,
        "device_type": device,
        "traffic_source": traffic,
        "hour_of_day": hour_of_day,
        "day_of_week": day_of_week,
        "previous_purchases_count": previous_purchases,
        "avg_time_per_page_seconds": avg_time_per_page,
        "added_to_wishlist": added_to_wishlist,
        "used_search": used_search,
    }

    if include_target:
        prob = 0.12
        if is_returning: prob += 0.22
        if has_discount_code: prob += 0.15
        if previous_purchases > 5: prob += 0.12
        if used_search: prob += 0.08
        if 60 <= session_duration_seconds <= 600: prob += 0.10
        if items_in_cart >= 2: prob += 0.08
        if traffic in ("email", "direct"): prob += 0.08
        if traffic == "paid": prob -= 0.08
        if added_to_wishlist: prob -= 0.10
        if device == "mobile" and session_duration_seconds < 60: prob -= 0.08
        if cart_total_value > 500 and is_returning == 0: prob -= 0.05
        prob = max(0.02, min(0.95, prob))
        session["converted"] = 1 if random.random() < prob else 0

    return session


def write_csv_dict(path: Path, rows: list) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the SAME preprocessing as model-training-code.py so the unseen
    CSV is already in the model's expected column order."""
    df = df.drop(columns=["session_id"])
    cols = ["device_type", "traffic_source"]
    for col in cols:
        df[col] = df[col].astype("category")
    for col in cols:
        df = pd.concat([df, pd.get_dummies(df[col], prefix=col, drop_first=True).astype(int)], axis=1)
        df = df.drop([col], axis=1)
    return df


def main() -> None:
    train_rows = [generate_session(f"CART_{i:05d}", include_target=True) for i in range(1, 20001)]
    unseen_rows = [generate_session(f"NEW_{i:05d}", include_target=False) for i in range(1, 101)]

    write_csv_dict(OUTPUT_DIR / "cart_sessions_train.csv", train_rows)
    print(f"Wrote {len(train_rows)} training rows (raw, with 'converted' target).")

    unseen_df = pd.DataFrame(unseen_rows)
    encoded = preprocess(unseen_df)
    encoded.to_csv(OUTPUT_DIR / "cart_sessions_new.csv", index=False)
    print(f"Wrote {len(encoded)} unseen rows (pre-encoded, {len(encoded.columns)} columns, no target).")
    print("Encoded columns:", list(encoded.columns))


if __name__ == "__main__":
    main()
