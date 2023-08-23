# Create a New Service Account in GCP before executing the below gcloud commands    
gcloud projects add-iam-policy-binding udemy-mlops \
    --member=serviceAccount:vertexai-sa@udemy-mlops.iam.gserviceaccount.com \
    --role=roles/aiplatform.customCodeServiceAgent

gcloud projects add-iam-policy-binding udemy-mlops \
    --member=serviceAccount:vertexai-sa@udemy-mlops.iam.gserviceaccount.com \
    --role=roles/aiplatform.admin

gcloud projects add-iam-policy-binding udemy-mlops \
    --member=serviceAccount:vertexai-sa@udemy-mlops.iam.gserviceaccount.com \
    --role=roles/storage.objectAdmin
