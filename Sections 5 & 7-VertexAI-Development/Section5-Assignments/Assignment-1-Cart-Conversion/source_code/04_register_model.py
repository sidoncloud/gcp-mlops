"""
04_register_model.py

Uploads (registers) the trained model.joblib artifact into the Vertex AI Model
Registry with a serving container image. Once registered the model can be
deployed to an online endpoint.

Run:
    python 04_register_model.py \
        --project $PROJECT_ID \
        --bucket  $BUCKET_NAME \
        --location us-central1
"""
import argparse

from google.cloud import aiplatform


SERVE_CONTAINER = "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-6:latest"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--location", default="us-central1")
    parser.add_argument("--model-display-name", default="cart-conversion-classifier")
    args = parser.parse_args()

    artifact_uri = f"gs://{args.bucket}/training/model"

    aiplatform.init(project=args.project, location=args.location)

    print(f"Registering model from {artifact_uri} ...")
    model = aiplatform.Model.upload(
        display_name=args.model_display_name,
        artifact_uri=artifact_uri,
        serving_container_image_uri=SERVE_CONTAINER,
        sync=True,
    )

    print("Model registered in Vertex AI Model Registry.")
    print(f"  Resource name: {model.resource_name}")
    print(f"  Display name:  {model.display_name}")
    print(f"  Version:       {model.version_id}")


if __name__ == "__main__":
    main()
