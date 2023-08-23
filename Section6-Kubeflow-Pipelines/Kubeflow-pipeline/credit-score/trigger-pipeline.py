from google.cloud import aiplatform

PROJECT_ID = "udemy-mlops"
REGION = "us-central1"
aiplatform.init(project=PROJECT_ID,location=REGION)

job = aiplatform.PipelineJob(
    display_name='trigger-credit-scoring-pipeline',
    template_path="gs://sid-kubeflow-v1/compiled_pipelines/credit-scoring-training.json",
    pipeline_root="gs://sid-kubeflow-v1/credit-scoring-pipeline",
    enable_caching=False
)
job.submit()