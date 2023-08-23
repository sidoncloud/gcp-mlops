from flask import Flask
from flask import request
import os 
from vertexai.language_models import TextGenerationModel

app = Flask(__name__)
generation_model = TextGenerationModel.from_pretrained("text-bison@001")

@app.route('/simple_classification',methods=['POST'])
def simple_classification():
    input_request_json = request.get_json()
    input_txt = input_request_json["msg"]
    prompt = f"""
        Given a piece of text, classify it as toxic or non-toxic. \n
        text: {input_txt}
    """
    response = generation_model.predict(
        prompt=prompt,
        max_output_tokens=256,
        temperature=0.1,
    ).text
    return {"response":response}

@app.route('/simple_classification_with_exp',methods=['POST'])
def classification_with_exp():
    input_request_json = request.get_json()
    input_txt = input_request_json["msg"]
    prompt = f"""
        Given a piece of text, classify it as toxic or non-toxic and explain why. \n
        text: {input_txt}
    """
    response = generation_model.predict(
        prompt=prompt,
        max_output_tokens=256,
        temperature=0.1,
    ).text    
    return {"response":response}

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5052)))