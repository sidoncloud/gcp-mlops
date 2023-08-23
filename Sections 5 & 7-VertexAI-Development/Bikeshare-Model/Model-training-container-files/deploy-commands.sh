
# Step-1 - Build the image 
docker build -t vertex-bikeshare-model .

# Step-2 - Tag the image locally
docker tag vertex-bikeshare-model gcr.io/udemy-mlops/vertex-bikeshare-model

# Step-3 - Push the image to Google Cloud Registry 
docker push gcr.io/udemy-mlops/vertex-bikeshare-model

# Step-4 - Submit a custom model training job  
gcloud ai custom-jobs create --region=us-central1 \
--project=udemy-mlops \
--worker-pool-spec=replica-count=1,machine-type='n1-standard-4',container-image-uri='gcr.io/udemy-mlops/vertex-bikeshare-model' \
--display-name=bike-sharing-model-training