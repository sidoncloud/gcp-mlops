# Step-1
docker build -t demo-flask-app .

# Push to Container Registry 
docker tag demo-flask-app gcr.io/udemy-mlops-395416/demo-flask-app
docker push gcr.io/udemy-mlops-395416/demo-flask-app

gcloud run deploy demo-flask-app --image gcr.io/udemy-mlops-395416/demo-flask-app --region us-central1


# Push to Artifact Registry 
docker tag demo-flask-app us-central1-docker.pkg.dev/udemy-mlops-395416/python-apps/demo-flask-app
docker push us-central1-docker.pkg.dev/udemy-mlops-395416/python-apps/demo-flask-app

gcloud run deploy demo-flask-app2 \
--image us-central1-docker.pkg.dev/udemy-mlops-395416/python-apps/demo-flask-app \
--region us-central1