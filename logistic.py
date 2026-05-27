import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

#data reading and encoding 
data=pd.read_csv(r"C:\flutter_projects\INTERNSHIP\class4\assignment2\data.csv")

label_encoder=LabelEncoder()
data["diagnosis"]=label_encoder.fit_transform(data["diagnosis"])
x=data[[
    "id",
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
    "compactness_mean",
    "concavity_mean",
    "concave points_mean",
    "symmetry_mean",
    "fractal_dimension_mean",
    "radius_se",
    "texture_se",
    "perimeter_se",
    "area_se",
    "smoothness_se",
    "compactness_se",
    "concavity_se",
    "concave points_se",
    "symmetry_se",
    "fractal_dimension_se",
    "radius_worst",
    "texture_worst",
    "perimeter_worst",
    "area_worst",
    "smoothness_worst",
    "compactness_worst",
    "concavity_worst",
    "concave points_worst",
    "symmetry_worst",
    "fractal_dimension_worst"
]]
y=data["diagnosis"]

#training
model=LogisticRegression()#logistic regression model
model.fit(x,y)

#coefficeint and intercept printing
print("model coefficient=",model.coef_)
print("model intercept=",model.intercept_)
sample=[[844981,13,21.82,87.5,519.8,0.1273,0.1932,0.1859,0.09353,0.235,0.07389,0.3063,1.002,2.406,24.32,0.005731,0.03502,0.03553,0.01226,0.02143,0.003749,15.49,30.73,106.2,739.3,0.1703,0.5401,0.539,0.206,0.4378,0.1072]]
#sample from the data itself

#prediction
prediction=model.predict(sample)
print("Predicted value=",prediction)
print("Value range=",model.predict_proba(sample))

#Predict whether the cancer is: • Malignant • Benign
print("the cancer is:")

if prediction[0] == 1:
    print("Malignant")
else:
    print("Benign")