from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from diabetes_classification.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

## Route for a home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            HighBP=int(request.form.get('HighBP')),
            HighChol=int(request.form.get('HighChol')),
            CholCheck=int(request.form.get('CholCheck')),
            BMI=float(request.form.get('BMI')),
            Smoker=int(request.form.get('Smoker')),
            Stroke=int(request.form.get('Stroke')),
            HeartDiseaseorAttack=int(request.form.get('HeartDiseaseorAttack')),
            PhysActivity=int(request.form.get('PhysActivity')),
            Fruits=int(request.form.get('Fruits')),
            Veggies=int(request.form.get('Veggies')),
            HvyAlcoholConsump=int(request.form.get('HvyAlcoholConsump')),
            AnyHealthcare=int(request.form.get('AnyHealthcare')),
            NoDocbcCost=int(request.form.get('NoDocbcCost')),
            GenHlth=int(request.form.get('GenHlth')),
            MentHlth=int(request.form.get('MentHlth')),
            PhysHlth=int(request.form.get('PhysHlth')),
            DiffWalk=int(request.form.get('DiffWalk')),
            Sex=int(request.form.get('Sex')),
            Age=int(request.form.get('Age')),
            Education=int(request.form.get('Education')),
            Income=int(request.form.get('Income'))
        )

        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline = PredictPipeline()
        print("Mid Prediction")
        results = predict_pipeline.predict(pred_df)
        print("after Prediction")
        if results[0] == 0 :
            results = 'Low'
        else:
            results = 'High'
        
        return render_template('home.html', results=results)

if __name__ == "__main__":
    app.run(host="0.0.0.0")
