from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict import CustomData, PredictPipeline

application = Flask(__name__)
app = application

# Route for home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        date_str = request.form.get('Datetime')
        dt = pd.to_datetime(date_str)
        year = dt.year
        month = dt.month
        day = dt.day

        data = CustomData(
            City=request.form.get('City'),
            PM2_5=float(request.form.get('PM2.5')),
            PM10=float(request.form.get('PM10')),
            NO=float(request.form.get('NO')),
            NO2=float(request.form.get('NO2')),
            NOx=float(request.form.get('NOx')),
            NH3=float(request.form.get('NH3')),
            CO=float(request.form.get('CO')),
            SO2=float(request.form.get('SO2')),
            O3=float(request.form.get('O3')),
            AQI_Bucket=request.form.get('AQI_Bucket'),
            year=year,
            month=month,
            day=day
        )

        pred_df = data.get_data_as_dataframe()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template('home.html', results=results[0])



if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)