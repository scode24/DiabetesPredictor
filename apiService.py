from flask import Flask, request
import pandas as pd
from sklearn import svm

app = Flask(__name__)


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    jsonData = request.json
    input = [[
        jsonData['pregnancies'],
        jsonData['glucose'],
        jsonData['blood_pressure'],
        jsonData['skin_thickness'],
        jsonData['insulin'],
        jsonData['bmi'],
        jsonData['diabetes_pedigree_function'],
        jsonData['age'],
    ]]
    df = pd.read_csv('diabetes.csv')
    x = df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
    y = df['Outcome']
    svc = svm.SVC(kernel='linear')
    svc.fit(x, y)
    prediction = svc.predict(input)
    if(predict == [0]):
        return 'not_diabetic'
    return 'diabetic'


app.run(port=9000)
