# Prerequisite: Install Uvicorn, OpenAPI
# To Run the File: uvicorn fastapi_app:app --reload

from fastapi import FastAPI, UploadFile, File
import shutil

# CV Libraries
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials

app = FastAPI()

@app.get("/")
async def welcome():
    return "Welcome to my Mini Project: Sports Classification using Azure Custom Vision."

@app.post("/sports_classifier")
async def custom_vision_classifier(file: UploadFile = File(...)):
   
    with open(r"testing/temp1.jpg", "wb") as f:
        shutil.copyfileobj(file.file, f)

    ENDPOINT = "Enter prediction endpoint"
    prediction_key = "Enter prediction key"
    project_id = "Enter project id"
    publish_iteration_name = "Enter iteration name"

    prediction_credentials = ApiKeyCredentials(in_headers = {"Prediction-key": prediction_key})
    predictor = CustomVisionPredictionClient(ENDPOINT, prediction_credentials)

    with open(r"testing/temp1.jpg", "rb") as image_contents:
        results = predictor.classify_image(project_id, publish_iteration_name, image_contents.read())
    tags = []
    prob = []
    for prediction in results.predictions:
        tags.append(prediction.tag_name)
        prob.append(prediction.probability * 100)

    return tuple(map(list, zip(tags, prob)))



