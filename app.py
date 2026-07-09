from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app = application


@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data = CustomData( # This is where the input data is collected from the form and passed to the CustomData class
            gender=request.form.get('gender', ''),
            race_ethnicity=request.form.get('race_ethnicity', ''),
            parental_level_of_education=request.form.get('parental_level_of_education', ''),    
            lunch=request.form.get('lunch', ''),
            test_preparation_course=request.form.get('test_preparation_course', ''),
            reading_score=int(request.form.get('reading_score') or 0),
            writing_score=int(request.form.get('writing_score') or 0)

        )
        pred_df = data.get_data_as_dataframe()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template('home.html', results=results[0]) # This is where the prediction result is passed to the home.html template


if __name__=="__main__":
    app.run(host="0.0.0.0", debug=True)

# CustomData is a class that takes in the input data and converts it into a pandas dataframe. The get_data_as_dataframe method is called to convert the input data into a dataframe.