from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)
model = pickle.load(open('rdf.pkl', 'rb'))

@app.route('/', methods=["GET"])
def home():
    return render_template('home.html')

@app.route('/submit', methods=["POST"])
def submit():
    if request.method == "POST":
        input_feature = [int(x) for x in request.form.values()]
        input_feature = [np.array(input_feature)]
        names = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area']
        data = pd.DataFrame(input_feature, columns=names)
        prediction = model.predict(data)
        prediction = int(prediction)
        if prediction == 0:
            return render_template("output.html", result="Loan Will Not be Approval")
        else:
            return render_template("output.html", result="Loan Will  be Approval")
        
     
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, port=port)
