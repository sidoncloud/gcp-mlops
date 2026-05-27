"""
Dataset generator for LAB_ECOM_VERTEXAI_002 - Product Return Prediction.

Generates two CSV files:
  - orders_train.csv   (20000 rows, with 'returned' target column)
  - orders_new.csv     (100  rows, unseen new orders, NO target column)
                       These represent orders that just landed in GCS.
"""
import csv
import random
from pathlib import Path

random.seed(42)

OUTPUT_DIR = Path(__file__).parent

CATEGORIES = ["Clothing", "Electronics", "Home", "Sports", "Books", "Beauty", "Toys"]
CATEGORY_WEIGHTS = [0.28, 0.18, 0.15, 0.12, 0.10, 0.10, 0.07]

PAYMENT_METHODS = ["credit_card", "paypal", "debit_card", "apple_pay", "gift_card"]
PAYMENT_WEIGHTS = [0.50, 0.20, 0.15, 0.10, 0.05]


def generate_order(order_id: str, include_target: bool = True) -> dict:
    num_items = random.choices([1, 2, 3, 4, 5, 6, 7, 8], weights=[40, 25, 15, 10, 5, 2, 2, 1], k=1)[0]
    item_price = round(random.uniform(8.0, 1200.0), 2)
    discount_applied_percent = random.choices([0, 5, 10, 15, 20, 25, 30, 40, 50],
                                              weights=[35, 15, 15, 10, 10, 7, 5, 2, 1], k=1)[0]
    order_total = round(item_price * num_items * (1 - discount_applied_percent / 100), 2)
    shipping_days = random.choices([1, 2, 3, 4, 5, 6, 7, 10, 14], weights=[5, 20, 25, 20, 15, 8, 4, 2, 1], k=1)[0]
    category = random.choices(CATEGORIES, weights=CATEGORY_WEIGHTS, k=1)[0]
    product_avg_rating = round(random.uniform(2.0, 5.0), 1)
    is_first_purchase = 1 if random.random() < 0.30 else 0
    if is_first_purchase:
        customer_past_order_count = 0
        customer_past_return_rate = 0.0
        customer_tenure_days = random.randint(0, 30)
    else:
        customer_past_order_count = random.randint(1, 80)
        customer_past_return_rate = round(random.uniform(0.0, 0.6), 3)
        customer_tenure_days = random.randint(31, 1800)
    payment_method = random.choices(PAYMENT_METHODS, weights=PAYMENT_WEIGHTS, k=1)[0]
    used_size_guide = 1 if (category == "Clothing" and random.random() < 0.45) else 0
    if category != "Clothing":
        used_size_guide = 0
    promo_used = 1 if discount_applied_percent >= 10 else 0
    weekend_order = 1 if random.random() < 0.30 else 0

    order = {
        "order_id": order_id,
        "order_total": order_total,
        "num_items": num_items,
        "item_price": item_price,
        "discount_applied_percent": discount_applied_percent,
        "shipping_days": shipping_days,
        "product_category": category,
        "product_avg_rating": product_avg_rating,
        "customer_past_order_count": customer_past_order_count,
        "customer_past_return_rate": customer_past_return_rate,
        "customer_tenure_days": customer_tenure_days,
        "payment_method": payment_method,
        "is_first_purchase": is_first_purchase,
        "used_size_guide": used_size_guide,
        "promo_used": promo_used,
        "weekend_order": weekend_order,
    }

    if include_target:
        prob = 0.12
        if category == "Clothing":
            prob += 0.22
        elif category == "Beauty":
            prob += 0.10
        elif category == "Books":
            prob -= 0.08
        elif category == "Electronics":
            prob += 0.05
        prob += min(0.25, customer_past_return_rate * 0.50)
        if discount_applied_percent >= 30:
            prob += 0.12
        if product_avg_rating < 3.5:
            prob += 0.10
        if is_first_purchase:
            prob += 0.08
        if used_size_guide == 1 and category == "Clothing":
            prob -= 0.12
        if shipping_days >= 7:
            prob += 0.05
        if order_total > 600:
            prob += 0.04
        prob = max(0.03, min(0.95, prob))
        order["returned"] = 1 if random.random() < prob else 0

    return order


def write_csv(path: Path, rows: list) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    train = [generate_order(f"ORD_{i:05d}", include_target=True) for i in range(1, 20001)]
    unseen = [generate_order(f"NEW_{i:05d}", include_target=False) for i in range(1, 101)]

    write_csv(OUTPUT_DIR / "orders_train.csv", train)
    write_csv(OUTPUT_DIR / "orders_new.csv", unseen)

    pos = sum(1 for r in train if r["returned"] == 1)
    print(f"Generated {len(train)} training rows ({pos} returned, {pos/len(train):.2%}).")
    print(f"Generated {len(unseen)} unseen rows (no target).")


if __name__ == "__main__":
    main()
