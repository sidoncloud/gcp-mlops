gcloud builds submit --region us-central1

  # Assign the roles 
  gcloud projects add-iam-policy-binding gcp-serverless-project-374110 \
  --member=serviceAccount:131640033627@cloudbuild.gserviceaccount.com --role=roles/iam.serviceAccountUser

  gcloud projects add-iam-policy-binding gcp-serverless-project-374110 \
  --member=serviceAccount:131640033627@cloudbuild.gserviceaccount.com --role=roles/run.admin