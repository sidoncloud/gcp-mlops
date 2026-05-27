"""
03_submit_training.py

Launches a Vertex AI Custom Training Job that runs trainer/task.py inside the
pre-built scikit-learn training container. Training artifacts are written to
gs://{bucket}/training/model/.

Run:
    python 03_submit_training.py \
        --project $PROJECT_ID \
        --bucket  $BUCKET_NAME \
        --location us-central1
"""
import argparse
import os

from google.cloud import aiplatform


TRAIN_CONTAINER = "us-docker.pkg.dev/vertex-ai/training/sklearn-cpu.1-6:latest"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--bucket", required=True, help="GCS bucket name, no gs:// prefix")
    parser.add_argument("--location", default="us-central1")
    parser.add_argument("--display-name", default="order-return-training")
    parser.add_argument("--machine-type", default="n1-standard-4")
    args = parser.parse_args()

    staging_uri = f"gs://{args.bucket}"
    base_output_dir = f"gs://{args.bucket}/training"
    training_uri = f"gs://{args.bucket}/data/orders_train.csv"

    print(f"Project:        {args.project}")
    print(f"Location:       {args.location}")
    print(f"Staging bucket: {staging_uri}")
    print(f"Output dir:     {base_output_dir}")
    print(f"Training URI:   {training_uri}")

    aiplatform.init(project=args.project, location=args.location, staging_bucket=staging_uri)

    script_path = os.path.join(os.path.dirname(__file__), "trainer", "task.py")

    job = aiplatform.CustomTrainingJob(
        display_name=args.display_name,
        script_path=script_path,
        container_uri=TRAIN_CONTAINER,
        requirements=[
            "pandas",
            "joblib",
            "google-cloud-storage",
            "gcsfs",
        ],
    )

    job.run(
        replica_count=1,
        machine_type=args.machine_type,
        args=[f"--training-data-uri={training_uri}"],
        base_output_dir=base_output_dir,
        sync=True,
    )

    print("Training job finished.")
    print(f"Model artifact directory: {base_output_dir}/model/")


if __name__ == "__main__":
    main()
