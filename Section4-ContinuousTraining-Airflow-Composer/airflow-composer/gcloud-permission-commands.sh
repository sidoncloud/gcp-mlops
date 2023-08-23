# Assign storage permissions to the service account 
gcloud projects add-iam-policy-binding udemy-mlops \
--member=serviceAccount:1090925531874-compute@developer.gserviceaccount.com --role=roles/storage.admin


gcloud projects add-iam-policy-binding udemy-mlops \
--member=serviceAccount:1090925531874-compute@developer.gserviceaccount.com --role=roles/storage.objects.list