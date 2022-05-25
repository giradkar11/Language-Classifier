import pickle
from flask import Flask, request, jsonify
import artifacts
import pandas as pd
import numpy as np

with open(r'artifacts\MultinominalNB.pkl','rb') as f:
    model = pickle.load(f)

with open(r'artifacts\tfidfvectorize.pkl','rb') as f:
    vectoriz = pickle.load(f)


app = Flask(__name__)

@app.route('/')
def home():
    return 'welcome to text classification API'

@app.route('/classifier' , methods=['POST','GET'])
def classifier():
    if request.method == 'POST':
        data = request.form
        print('*'*60)
        print(data)
        print('*'*60)

        input_text = data['text']

        vectoriz_train = vectoriz.transform([input_text])
        print(vectoriz_train.A)
        print(vectoriz.get_feature_names_out())

        df_train = pd.DataFrame(vectoriz_train.A , columns= vectoriz.get_feature_names_out())
        prediction = model.predict(df_train)

        print(prediction)

        return jsonify( {'language' : prediction[0]})


if __name__=='__main__':
    app.run(debug = False)