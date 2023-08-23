FROM python:3.10-slim
WORKDIR /

COPY model_training_code.py model_training_code.py
COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt
ENTRYPOINT ["python3","model_training_code.py"]