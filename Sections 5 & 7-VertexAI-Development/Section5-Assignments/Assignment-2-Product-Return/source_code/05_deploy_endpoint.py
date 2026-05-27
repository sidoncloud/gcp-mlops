"""
05_deploy_endpoint.py

Creates a Vertex AI Endpoint and deploys the most recent version of the
registered model to it for real-time online predictions.

Run:
    python 05_deploy_endpoint.py \
        --project $PROJECT_ID \
        --location us-central1
"""
import argparse

from google.cloud import aiplatform


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--location", default="us-central1")
    parser.add_argument("--model-display-name", default="order-return-classifier")
    parser.add_argument("--endpoint-display-name", default="order-return-endpoint")
    parser.add_argument("--machine-type", default="n1-standard-2")
    parser.add_argument("--min-replicas", type=int, default=1)
    parser.add_argument("--max-replicas", type=int, default=2)
    args = parser.parse_args()

    aiplatform.init(project=args.project, location=args.location)

    models = aiplatform.Model.list(filter=f'display_name="{args.model_display_name}"')
    if not models:
        raise SystemExit(f"No model found with display_name={args.model_display_name}")
    model = models[0]
    print(f"Using model: {model.resource_name}")

    endpoints = aiplatform.Endpoint.list(filter=f'display_name="{args.endpoint_display_name}"')
    if endpoints:
        endpoint = endpoints[0]
        print(f"Reusing endpoint: {endpoint.resource_name}")
    else:
        endpoint = aiplatform.Endpoint.create(display_name=args.endpoint_display_name)
        print(f"Created endpoint: {endpoint.resource_name}")

    print("Deploying model to endpoint (this takes 5 to 10 minutes)...")
    endpoint.deploy(
        model=model,
        deployed_model_display_name=args.model_display_name,
        machine_type=args.machine_type,
        min_replica_count=args.min_replicas,
        max_replica_count=args.max_replicas,
        traffic_percentage=100,
        sync=True,
    )

    endpoint_id = endpoint.resource_name.split("/")[-1]
    print("\nDeployment complete.")
    print(f"ENDPOINT_ID={endpoint_id}")
    print(f"ENDPOINT_RESOURCE={endpoint.resource_name}")
    print("\nExport this in your shell so the predict script can use it:")
    print(f"  export ENDPOINT_ID={endpoint_id}")


if __name__ == "__main__":
    main()
