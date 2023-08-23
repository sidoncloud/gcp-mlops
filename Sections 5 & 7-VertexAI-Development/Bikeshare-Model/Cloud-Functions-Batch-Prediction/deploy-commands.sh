# APIs to be enabled 
- cloud function 
- cloud build 
- eventarc

# gcloud projects add-iam-policy-binding 1090925531874 \
#   --member serviceAccount:1090925531874-compute@developer.gserviceaccount.com  \
#   --role='roles/aiplatform.admin'
  

gcloud functions deploy bikeshare-batch-run \
--runtime=python310 \
--region=us-central1 \
--source=. \
--entry-point=trigger_batch_predictions \
--trigger-resource=sid-bikeshare-new-data \
--trigger-event=google.storage.object.finalize