"""
99_cleanup.py

Tear down all Vertex AI resources this lab created so there are no ongoing
charges: undeploys and deletes the endpoint, deletes all model versions,
then deletes the GCS bucket.

Run:
    python 99_cleanup.py --project $PROJECT_ID --bucket $BUCKET_NAME --location us-central1
"""
import argparse

from google.cloud import aiplatform, storage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--location", default="us-central1")
    parser.add_argument("--model-display-name", default="order-return-classifier")
    parser.add_argument("--endpoint-display-name", default="order-return-endpoint")
    parser.add_argument("--delete-bucket", action="store_true", help="Also delete the GCS bucket")
    args = parser.parse_args()

    aiplatform.init(project=args.project, location=args.location)

    for endpoint in aiplatform.Endpoint.list(filter=f'display_name="{args.endpoint_display_name}"'):
        print(f"Undeploying and deleting endpoint {endpoint.resource_name}")
        endpoint.undeploy_all()
        endpoint.delete()

    for model in aiplatform.Model.list(filter=f'display_name="{args.model_display_name}"'):
        print(f"Deleting model {model.resource_name}")
        model.delete()

    if args.delete_bucket:
        print(f"Deleting bucket gs://{args.bucket}")
        client = storage.Client(project=args.project)
        bucket = client.bucket(args.bucket)
        bucket.delete(force=True)
    else:
        print(f"Leaving bucket gs://{args.bucket} in place. "
              "Re-run with --delete-bucket to remove it.")

    print("Cleanup complete.")


if __name__ == "__main__":
    main()
