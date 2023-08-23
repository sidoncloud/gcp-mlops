docker build -t roi_model_serving .

docker tag roi_model_serving gcr.io/udemy-mlops/roi_model_serving

docker push gcr.io/udemy-mlops/roi_model_serving

gcloud run deploy roi-model-inference --image  gcr.io/udemy-mlops/roi_model_serving --region us-central1

