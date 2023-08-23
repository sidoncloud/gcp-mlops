# Submit Model Training Job to Vertex AI 
gcloud ai custom-jobs create --region=us-central1 \
--project=udemy-mlops \
--worker-pool-spec=replica-count=1,machine-type='n1-standard-4',container-image-uri='gcr.io/udemy-mlops/cicd-vertex-bikeshare-model' \
--display-name=bike-sharing-model-training

# Upload Trained Model to Vertex AI Model Registry 
gcloud ai models upload \
--container-image-uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest" \
--description=bikeshare-new-model --display-name=bikeshare-new-model \
--artifact-uri='gs://sid-ml-ops/bike-share-rf-regression-artifact/' \
--project=udemy-mlops --region=us-central1

# Deploy Model to the Endpoint
gcloud beta ai endpoints deploy-model $ENDPOINT_ID \
  --region=us-central1 \
  --model=$MODEL_ID \
  --display-name=bikeshare-model-endpoint \
  --traffic-split=0=100 \
  --machine-type=n1-standard-4