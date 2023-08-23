curl -X POST https://llm-text-classifcation-ucinc65roa-uc.a.run.app/simple_classification \
-H "Content-Type: application/json" \
-d '{
     "msg":"Im wondering where to travel next"
}'

curl -X POST https://llm-text-classifcation-ucinc65roa-uc.a.run.app/simple_classification_with_exp \
-H "Content-Type: application/json" \
-d '{
     "msg":"Im wondering where to travel next"
}'

curl -X POST https://llm-text-classifcation-ucinc65roa-uc.a.run.app/simple_classification_with_exp \
-H "Content-Type: application/json" \
-d '{
     "msg":"Im wondering if i should invest in the startup"
}'

curl -X POST https://llm-text-classifcation-ucinc65roa-uc.a.run.app/simple_classification_with_exp \
-H "Content-Type: application/json" \
-d '{
     "msg":"i was upset with the performance ratings after having worked so hard"
}'
