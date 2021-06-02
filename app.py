# https://www.tutorialspoint.com/flask
from flask import Flask, render_template, request
import numpy as np

import joblib

import flask
app = Flask(__name__)




@app.route('/')
def home():
    return flask.render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    #to_predict = request.form.values()
    input=request.form.to_dict()
    value = list(input.values())

    to_predict_list = np.array(value)
    category_cols=to_predict_list[np.array([0,13,14])]
   # return render_template("index.html", prediction=str(category_cols))
    xq_point_df = to_predict_list[np.array([1,2,3,4,5,6,7,8,9,10,11,12])]


    ce_ohe_cat = joblib.load('sklearn_ohe.pkl')


    xq_list_new2 = ce_ohe_cat.transform(category_cols.reshape(1,-1))
    return render_template("index.html", prediction=str(category_cols.reshape(1,-1)))

    autoscaler = joblib.load('autoscaler.pkl')
    xq_point_new = autoscaler.transform(xq_point_df)

    lgbm_model = joblib.load('lgbm_model.pkl')
    x_predit=[xq_list_new2[0]].extend(xq_point_new).extend(xq_list_new2[1,2])


    y_pred = lgbm_model.predict(x_predit.reshape(1,-1))
    print('Predicted Class for xq point: ',y_pred)
    class_dict={1:'CLASS-1-LOW',2:'CLASS-2-MEDIUM',3:'CLASS-3-HIGH',4:'CLASS-4-VERY_HIGH'}

            
    return render_template("index.html", prediction = class_dict[y_pred])



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
