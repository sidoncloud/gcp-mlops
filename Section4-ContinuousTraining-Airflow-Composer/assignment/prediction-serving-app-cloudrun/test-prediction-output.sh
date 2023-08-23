# Test using below CMD after running the flask app locally 
curl -X POST http://127.0.0.1:5050/predict \
-H "Content-Type: application/json" \
-d '{"index":12,"EMAIL":521698,"SEARCH_ENGINE":521339,"SOCIAL_MEDIA":521528,"VIDEO":519625}'

curl -X POST http://127.0.0.1:5050/predict \
-H "Content-Type: application/json" \
-d '{"index":56,"EMAIL":520742,"SEARCH_ENGINE":522807,"SOCIAL_MEDIA":518987,"VIDEO":521946}'

# Test using below CMD after deploying the app to Cloud-Run
curl -X POST https://roi-model-inference-ucinc65roa-uc.a.run.app/predict \
-H "Content-Type: application/json" \
-d '{"index":56,"EMAIL":520742,"SEARCH_ENGINE":522807,"SOCIAL_MEDIA":518987,"VIDEO":521946}'