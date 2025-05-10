from flask import Flask, request, render_template
import pandas as pd
import pickle

# Load model pipeline
with open('pipeline.pkl', 'rb') as f:
    pipeline = pickle.load(f)

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        try:
            gender = request.form['Gender']
            age = float(request.form['Age'])
            height = float(request.form['Height'])
            weight = float(request.form['Weight'])
            duration = float(request.form['Duration'])
            heart_rate = float(request.form['Heart_Rate'])
            body_temp = float(request.form['Body_Temp'])

            # Create sample DataFrame
            sample = pd.DataFrame([{
                'Gender': gender,
                'Age': age,
                'Height': height,
                'Weight': weight,
                'Duration': duration,
                'Heart_Rate': heart_rate,
                'Body_Temp': body_temp
            }])

            # Predict
            prediction = round(pipeline.predict(sample)[0], 2)

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
