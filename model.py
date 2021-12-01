import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import pickle
data = pd.read_csv('insurance_claims.csv')

data.replace('?', np.nan, inplace = True)

data['collision_type'] = data['collision_type'].fillna(data['collision_type'].mode()[0])

data['property_damage'] = data['property_damage'].fillna(data['property_damage'].mode()[0])

data['police_report_available'] = data['police_report_available'].fillna(data['police_report_available'].mode()[0])

data['policy_annual_premium'] = data['policy_annual_premium'].astype(int)

to_drop = ['policy_number','policy_bind_date','policy_state','insured_zip','incident_location','incident_date',
           'incident_state','incident_city','insured_hobbies','auto_model', '_c39', 'insured_occupation', "capital-gains", "capital-loss", 'insured_relationship', 'policy_csl']

data.drop(to_drop, inplace = True, axis = 1) 
data.drop(columns = ['total_claim_amount'], inplace = True, axis = 1)
data.drop(columns = ['injury_claim', 'vehicle_claim', 'property_claim', 'policy_deductable', 'policy_annual_premium', 'umbrella_limit'], inplace = True, axis = 1)
X = data.drop('fraud_reported', axis = 1)
y = data['fraud_reported']

cat_df = X.select_dtypes(include = ['object'])


cleanup_nums = {"insured_sex": {"MALE": 1, "FEMALE": 2},
                "insured_education_level": {"JD": 5,"PhD": 4,"Associate": 7,"MD": 6,"High School": 1,"Masters": 3,"College": 2},
                "auto_make": {"Saab": 13,"Dodge": 12,"Suburu": 6,"Nissan": 3,"Chevrolet": 4,"Ford": 10,"BMW": 7,"Toyota": 1,"Audi": 9,"Volkswagen": 11,"Accura": 5,"Jeep": 14,"Mercedes": 8,"Honda": 2},
                "incident_severity": {"Minor Damage": 1,"Major Damage": 2,"Total Loss": 3,"Trivial Damage": 4},
                "collision_type": {"Rear Collision": 1,"Side Collision": 2,"Front Collision": 3},
                "incident_type": {"Multi-vehicle Collision": 1,"Single Vehicle Collision": 2,"Vehicle Theft": 3,"Parked Car": 4},
                "property_damage": {"NO": 1,"YES": 2},
                "authorities_contacted": {"Police": 1,"Fire": 2,"Other": 3,"Ambulance": 4,"None": 5},
                "police_report_available": {"NO": 1,"YES": 2}}

cat_df = cat_df.replace(cleanup_nums)

num_df = X.select_dtypes(include = ['int'])
X = pd.concat([num_df, cat_df], axis = 1)
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
x_resampled, y_resampled = SMOTE().fit_resample(X, y)
print(x_resampled.count())
X_train, X_test, y_train, y_test = train_test_split(x_resampled, y_resampled,test_size = 0.25, random_state = 42)
num_df = X_train[['age', 'months_as_customer', 'incident_hour_of_the_day',
       'number_of_vehicles_involved', 'bodily_injuries', 'witnesses']]

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()

rand_grid = {"bootstrap": [True, False],
             "n_estimators": [200, 800, 1000],
             "max_depth": [70,90,None],
             "max_features": ["auto","sqrt"],
             "min_samples_split": [4, 5, 7],
             "min_samples_leaf": [1, 2]}

            
gs_clf = GridSearchCV(estimator = model, param_grid = rand_grid, cv = 5)
gs_clf.fit(X_train.values, y_train)
y_preds = gs_clf.predict(X_test.values)
print(classification_report(y_test, y_preds))
print(confusion_matrix(y_test, y_preds))
pickle.dump(gs_clf, open('model.pkl', 'wb'))
