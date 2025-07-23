import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split 

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix , f1_score , recall_score , precision_score

from sklearn.preprocessing import LabelEncoder

import joblib

data = pd.read_csv("lungs.csv")


#print(data.head())


# yar hum pahlay ununcessery items ko drop karay ga 

data.drop (['index' , 'Patient Id'], axis=1, inplace=True) 

# yar ab hum class label ko numeeic data ma convetr karay ga 

le = LabelEncoder()
data['Level'] = le.fit_transform(data['Level'])

print(le.classes_)

#print(data.head())

# data preprocessing check for missing values 

print(data.isnull().sum())

# ab target batoo 

x = data.drop('Level', axis=1)
y = data['Level'] 

# split the data into training and testing sets
x_train,x_test,y_train , y_test = train_test_split(x,y,test_size=0.2,random_state=42)

# Initialize the model

model = RandomForestClassifier()
model.fit(x_train , y_train)

# make predictions
y_predict = model.predict(x_test)

# print the accuracy of the model
print("accuray" , accuracy_score(y_test , y_predict ))


print("classification report",classification_report(y_test,y_predict))

# Step 8: Plot feature importance
plt.figure(figsize=(10, 6))
feat_importance = pd.Series(model.feature_importances_, index=x.columns)
feat_importance.nlargest(10).plot(kind='barh')
plt.title("Important Features for Heart Disease Prediction")
plt.show()

joblib.dump(model, "lungs_model.pkl")
joblib.dump(x.columns.tolist(), "lungs_model_features.pkl")
print("Model and features saved successfully.")