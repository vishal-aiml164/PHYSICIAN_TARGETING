import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import LGBMModel,LGBMClassifier
import pickle
from pickle import dump
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import joblib
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split


all_data_new = pd.read_csv('all_data_new.csv')
all_data_new.drop(['saving_cards_dropped','vouchers_dropped','total_seminar_as_attendee','total_seminar_as_speaker'
,'total_prescriptions_for_indication3','brand_web_impressions','brand_ehr_impressions','brand_enews_impressions','brand_mobile_impressions'
,'brand_organic_web_visits','brand_paidsearch_visits','urban_population_perc_in_physician_locality','percent_population_with_health_insurance_in_last10q'
,'physician_tenure','physician_age'], axis = 1,inplace = True)
CLASS_LABEL ='physician_segment_ordinal'
print(f'\nTotal Dataset contains {all_data_new.shape[0]} samples and {all_data_new.shape[1]} variables')
features = [c for c in all_data_new.columns if c not in [CLASS_LABEL]]
print(f'\nThe dataset contains {len(features)} features and 1 CLASS LABEL')
category_cols= ['physician_gender', 'physician_speciality', 'year_quarter']
numerical_cols = [c for c in features if c not in category_cols]

y_class = all_data_new[CLASS_LABEL]
CLASS_LABEL ='physician_segment_ordinal'
# split the data into test and train by maintaining same distribution of output varaible 'y_true' [stratify=y_true]
X_train, X_test, y_train, y_test = train_test_split(all_data_new.drop(['physician_segment_ordinal'], axis=1), y_class,stratify=y_class,test_size=0.20)
# split the train data into train and cross validation by maintaining same distribution of output varaible 'y_train' [stratify=y_train]
X_train, X_cv, y_train, y_cv = train_test_split(X_train, y_train,stratify=y_train,test_size=0.20)


sklearn_ohe = preprocessing.OneHotEncoder()
X_train_ohe = sklearn_ohe.fit_transform(X_train[category_cols])
joblib.dump(sklearn_ohe, 'sklearn_ohe.pkl')
imported_sklearn_ohe = joblib.load('sklearn_ohe.pkl')
X_cv_ohe = imported_sklearn_ohe.transform(X_cv[category_cols])
X_test_ohe = imported_sklearn_ohe.transform(X_test[category_cols])

X_train_ohe_df = pd.DataFrame(data = X_train_ohe.toarray())
X_cv_ohe_df = pd.DataFrame(data = X_cv_ohe.toarray())    
X_test_ohe_df = pd.DataFrame(data = X_test_ohe.toarray())     

X_train.reset_index(drop=True, inplace=True)
X_train_ohe_df.reset_index(drop=True, inplace=True)
train_final=pd.concat([X_train, X_train_ohe_df], axis=1)

X_cv.reset_index(drop=True, inplace=True)
X_cv_ohe_df.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)
X_test_ohe_df.reset_index(drop=True, inplace=True)
cv_final=pd.concat([X_cv, X_cv_ohe_df], axis=1)
test_final=pd.concat([X_test, X_test_ohe_df], axis=1)          

train_final.drop(['year_quarter','physician_gender','physician_speciality'], axis = 1,inplace = True)
cv_final.drop(['year_quarter','physician_gender','physician_speciality'], axis = 1,inplace = True)
test_final.drop(['year_quarter','physician_gender','physician_speciality'], axis = 1,inplace = True)

autoscaler = MinMaxScaler()
#autoscaler = StandardScaler()
X_train_ce = autoscaler.fit_transform(train_final)
autoscaler_filename = 'autoscaler.pkl'
joblib.dump(autoscaler, open(autoscaler_filename, 'wb'))     

autoscaler = joblib.load(open('autoscaler.pkl', 'rb'))
X_cv_ce = autoscaler.transform(cv_final)
X_test_ce = autoscaler.transform(test_final)

x_cfl=LGBMClassifier(n_estimators=2000, learning_rate=0.2, colsample_bytree=0.9, max_depth=10,subsample=0.7,nthread=-1)
x_cfl.fit(X_train_ce,y_train)
# save the model to disk
model_filename = 'lgbm_model.pkl'
joblib.dump(x_cfl, open(model_filename, 'wb'))

lgb_model = joblib.load(open('lgbm_model.pkl', 'rb'))


predict_y = lgb_model.predict_proba(X_train_ce)
print ('\nFor values of best params train loss',log_loss(y_train, predict_y))
predict_y = lgb_model.predict_proba(X_cv_ce)
print ('\nFor values of best params cv loss',log_loss(y_cv, predict_y))
predict_y = lgb_model.predict_proba(X_test_ce)
print ('\nFor values of best params test loss',log_loss(y_test, predict_y))
                 