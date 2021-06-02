# https://www.tutorialspoint.com/flask
from flask import Flask, jsonify, request
import numpy as np
#from sklearn.externals import joblib
import joblib
import pandas as pd
#from category_encoders import one_hot

import flask
app = Flask(__name__)




@app.route('/')
def home():
    return flask.render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    #lr = joblib.load('model.pkl')
    to_predict_list = request.form.to_dict()
    print(to_predict_list)
    to_predict_list = list(to_predict_list.values())
    print(to_predict_list)
    to_predict_list = np.array(list( to_predict_list)).reshape(1, -1)
    print(to_predict_list)
    print(len(to_predict_list))    
    print(type(to_predict_list))
    
    category_cols= ['physician_gender', 'physician_speciality', 'year_quarter']
    xq_point_df = pd.DataFrame(data=to_predict_list, columns=['year_quarter', 'brand_prescribed', 'total_representative_visits',
                                                   'total_sample_dropped', 'physician_hospital_affiliation', 'physician_in_group_practice',
                                                   'total_prescriptions_for_indication1', 'total_prescriptions_for_indication2', 'total_patient_with_commercial_insurance_plan',
                                                   'total_patient_with_medicare_insurance_plan', 'total_patient_with_medicaid_insurance_plan', 'total_competitor_prescription',
                                                   'new_prescriptions', 'physician_gender', 'physician_speciality',])
    print(xq_point_df)

    ce_ohe_cat = joblib.load('sklearn_ohe.pkl')
    #xq_point_new = ce_ohe_cat.transform(xq_list_new1)

    xq_list_new2 = ce_ohe_cat.transform(xq_point_df[category_cols])
    xq_list_new2_df = pd.DataFrame(data = xq_list_new2.toarray() )
                                         
    xq_point_df.reset_index(drop=True, inplace=True)
    xq_list_new2_df.reset_index(drop=True, inplace=True)

    xq_point_final=pd.concat([xq_point_df, xq_list_new2_df], axis=1)
    xq_point_final.drop(['year_quarter','physician_gender','physician_speciality'], axis = 1,inplace = True)


    autoscaler = joblib.load('autoscaler.pkl')
    xq_point_new = autoscaler.transform(xq_point_final)
    #rf_model = joblib.load('rf_model.pkl')
    lgbm_model = joblib.load('lgbm_model.pkl')
    #y_pred = rf_model.predict(xq_point_new)
    y_pred = lgbm_model.predict(xq_point_new)
    #y_pred_proba = lgbm_model.predict_proba(xq_point_new)
    print('Predicted Class for xq point: ',y_pred)
    if (y_pred == [1]):
        y_pred_new='CLASS-1-LOW'
    elif (y_pred == [2]):
        y_pred_new='CLASS-2-MEDIUM'
    elif (y_pred == [3]):
        y_pred_new='CLASS-3-HIGH'
    else: 
        y_pred_new='CLASS-4-VERY_HIGH'
    print('Predicted Text for xq point: ',y_pred_new)
            
    return jsonify({'prediction': y_pred_new})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
