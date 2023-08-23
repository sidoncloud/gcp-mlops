# Assign Service account user role to the service account 
gcloud projects add-iam-policy-binding udemy-mlops \
--member=serviceAccount:1090925531874@cloudbuild.gserviceaccount.com --role=roles/aiplatform.admin