# Assign Service account user role to the service account 
gcloud projects add-iam-policy-binding udemy-mlops \
--member=serviceAccount:1090925531874@cloudbuild.gserviceaccount.com --role=storage.buckets.get


# Assign Cloud Run role to the service account 
gcloud projects add-iam-policy-binding udemy-mlops \
  --member=serviceAccount:1090925531874@cloudbuild.gserviceaccount.com --role=roles/run.admin

# Command to run the build using cloudbuild.yaml
gcloud builds submit --region us-central1


