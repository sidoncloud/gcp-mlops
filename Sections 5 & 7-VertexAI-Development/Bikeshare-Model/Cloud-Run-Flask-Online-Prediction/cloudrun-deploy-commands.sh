# Grant the necessary permissions to the cloud run Service Account : 
gcloud projects add-iam-policy-binding 1090925531874 \
  --member serviceAccount:1090925531874-compute@developer.gserviceaccount.com  \
  --role='roles/aiplatform.admin'

# Step-1
docker build -t bikeshare-online-predict .
# Push to Container Registry 
docker tag bikeshare-online-predict gcr.io/udemy-mlops/bikeshare-online-predict
docker push gcr.io/udemy-mlops/bikeshare-online-predict

gcloud run deploy bikeshare-online-predict --image  gcr.io/udemy-mlops/bikeshare-online-predict --region us-central1