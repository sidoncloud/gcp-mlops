# Step-1
docker build -t llm-text-classifcation .

# Push to Container Registry 
docker tag demo-llm-flask gcr.io/udemy-mlops/llm-text-classifcation
docker push gcr.io/udemy-mlops/llm-text-classifcation

gcloud run deploy llm-text-classifcation --image gcr.io/udemy-mlops/llm-text-classifcation --region us-central1