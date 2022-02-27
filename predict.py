import pickle

from flask import Flask
from flask import request
from flask import jsonify


model_file = 'model_Credit.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

# Name the app
app = Flask('Default')

#put it in predict and send a JSON of customer details using post
@app.route('/predict', methods=['POST'])
def predict():
    #tell flask we getting json it will return a python dictionary
    customer = request.get_json()
    
    # turn the customer details to a feature matrix and invoking the model
    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]

    default = y_pred >= 0.5

    result = {
        'Default_probability': float(y_pred), #turn numpy float to python float
        'Default': bool(default) #turns the numpy bool to python bool 
    }
    # We convert the python dict and return a json
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)