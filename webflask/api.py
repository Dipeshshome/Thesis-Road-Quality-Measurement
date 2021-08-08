import io
import os
import json
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision import models
from flask import render_template
from PIL import Image
from flask import Flask, jsonify, request
from classes import caltech256Classes

app = Flask(__name__)
UPLOAD_FOLDER="/Users/Dipesh Shome/Desktop/webflask/static/"


def create_model(n_classes):
  model =  torch.hub.load('pytorch/vision', 'mobilenet_v2', pretrained=True)

  model.classifier[1] = torch.nn.Linear(in_features=model.classifier[1].in_features, out_features=n_classes)

  return model

 # Evaluation mode, IMPORTANT

def transform_image(image_bytes): 
    # We will recieve the image as bytes
    transform_image = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])

    #image = Image.open(io.BytesIO(image_bytes))
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return transform_image(image).unsqueeze(1)

def get_prediction(image_bytes):
    tensor = transform_image(image_bytes)
    outputs = model.forward(tensor.view(-1, 3,224,224))
    _,y_hat = outputs.max(1)
    predicted_idx = y_hat.item()
    print(predicted_idx)
    return caltech256Classes[predicted_idx]

@app.route('/', methods=['GET','POST'])
def predict():
    if request.method == "POST":
        image = request.files['image']
        img_bytes = image.read()
        class_name = get_prediction(img_bytes)
        if image:
            image_loaction=os.path.join(UPLOAD_FOLDER,image.filename)
            image.save(image_loaction)
            return render_template("index.html",prediction=class_name,image_loc=image.filename)
            
        return jsonify({
            "class_name":class_name
        })
    return render_template("index.html",prediction=0,image_loc=None)
    

if __name__ == '__main__':
    model = create_model(5)
    #model.classifier[-1]
    #IN_FEATURES = model.classifier[-1].in_features 
    #final_fc = nn.Linear(IN_FEATURES,5)
    #model.classifier[-1] = final_fc
    model.load_state_dict(torch.load("best_model_Mobilenet6040_224 (1).bin", map_location="cpu")) # Model download link is shown above too
    model.eval()
    app.run(port=12000,debug=True)