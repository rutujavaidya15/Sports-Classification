# Prerequisite: flask, werkzeug,
# To Run the File: python flask_app.py

import os
from flask import Flask, request
from werkzeug.utils import secure_filename

# CV Libraries
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials

UPLOAD_FOLDER = 'testing'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/", methods=['GET'])
def welcome():
    return "Welcome to my Mini Project: Sports Classification using Azure Custom Vision."

@app.route("/sports_classifier", methods=['POST'])
def custom_vision_classifier():
    if request.method=='POST':
        image=request.files['image']
        
        image_name = secure_filename(image.filename)
        image.save(os.path.join(app.config['UPLOAD_FOLDER'], image_name))
        image_url = os.path.join(app.config['UPLOAD_FOLDER'], image_name)

        ENDPOINT = "Enter prediction endpoint"
        prediction_key = "Enter prediction key"
        project_id = "Enter project id"
        publish_iteration_name = "Enter iteration name"

        prediction_credentials = ApiKeyCredentials(in_headers = {"Prediction-key": prediction_key})
        predictor = CustomVisionPredictionClient(ENDPOINT, prediction_credentials)

        with open(image_url, "rb") as image_contents:
            results = predictor.classify_image(project_id, publish_iteration_name, image_contents.read())
        
        tags = []
        prob = []
        for prediction in results.predictions:
            tags.append(prediction.tag_name)
            prob.append(prediction.probability * 100)
        
        result = {t:p for t,p in zip(tags, prob)}
    return result


if __name__=="__main__":
    app.run(debug=True)