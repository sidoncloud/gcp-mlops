from flask import Flask
from flask import request
import os 
from docx import Document
from io import BytesIO
from vertexai.language_models import TextGenerationModel
from google.cloud import storage

app = Flask(__name__)
generation_model = TextGenerationModel.from_pretrained("text-bison@001")
client = storage.Client()
bucket_name = 'sid-ml-ops'
bucket = client.bucket(bucket_name)

@app.route('/summarize_word_documents',methods=['POST'])
def summarize_word_documents():
    input_request_json = request.get_json()
    doc_file = input_request_json["file_name"]

    blob = bucket.blob("gen-ai/"+doc_file)
    word_file_in_bytes = blob.download_as_bytes()
    document = Document(BytesIO(word_file_in_bytes))

    text = []
    for paragraph in document.paragraphs:
        text.append(paragraph.text)
    
    full_text = '\n'.join(text)

    prompt = f"""
        Provide a very short summary, no more than 50 words, for the following article: \n
        text: {full_text}
    """
    response = generation_model.predict(
        prompt=prompt,
        max_output_tokens=256,
        temperature=0.1,
    ).text
    return {"response":response}

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5050)))