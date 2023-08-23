import functions_framework
from google.cloud import aiplatform
from google.cloud.aiplatform.gapic import JobState
import logging
import json

@functions_framework.cloud_event
def trigger_batch_predictions(cloud_event):

    project_id = "udemy-mlops"
    region = "us-central1"
    staging_bucket="gs://sid-vertex-mlops"

    aiplatform.init(project=project_id, location=region, staging_bucket=staging_bucket)

    gcs_input_uri = 'gs://sid-vertex-mlops/bike-share/batch.csv'
    BUCKET_URI = "gs://sid-vertex-mlops/bikeshare-batch-prediction-result/"
    model_id = 675603715779985408
    project_id = 1090925531874
    model = aiplatform.Model('projects/{}/locations/us-central1/models/{}'.format(project_id,model_id))

    batch_predict_job = model.batch_predict(
        job_display_name="bikeshare_batch_predict",
        gcs_source=gcs_input_uri,
        gcs_destination_prefix=BUCKET_URI,
        instances_format="csv",
        predictions_format="jsonl",
        machine_type="n1-standard-4",
        starting_replica_count=1,
        max_replica_count=1,
        sync=False
    )

    batch_predict_job.wait()

    if batch_predict_job.state == JobState.JOB_STATE_SUCCEEDED:
        logging.info(json.dumps({"message": "The job succeeded!", "job_id": batch_predict_job.name, "status": "success"}))
    else:
        logging.info(json.dumps({"message": "The job has not yet succeeded.", "job_id": batch_predict_job.name, "status": "pending"}))