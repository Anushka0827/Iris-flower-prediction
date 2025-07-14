from flask import Flask, render_template, request
import pickle
import numpy as np
# import webbrowser
# import threading

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    print("✅ Rendering index.html template")
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(request.form[x]) for x in ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    prediction = model.predict([features])[0]
    flower_map = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
    flower = flower_map[prediction]
    return render_template('index.html', prediction_text=f'Predicted Iris Species: {flower}')

# def open_browser():
#     webbrowser.open_new("http://127.0.0.1:5001")

if __name__ == "__main__":
    # threading.Timer(1.5, open_browser).start()
    from waitress import serve
    serve(app, host="127.0.0.1", port=5001)
